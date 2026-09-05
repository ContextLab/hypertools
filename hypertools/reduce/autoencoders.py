"""Torch-backed autoencoder reducers (GH #162).

Six scikit-learn-style dimensionality-reduction classes, each wrapping a
small `torch.nn.Module` autoencoder and its own training loop, registered
in `hypertools.reduce.common.REDUCERS` (resolved lazily -- see
`hypertools.reduce.common.resolve_reducer`, mirroring how `'UMAP'` is
resolved) so `import hypertools` never requires `torch` to be installed.
`torch` ships as the optional `[torch]` extra; this module is only ever
imported once a caller actually asks for one of the six names below, at
which point a missing `torch` is installed on demand
(`hypertools._shared.lazy_import`) and, only if that fails, surfaces as an
`ImportError` naming the manual command.

Variant selection guidance
---------------------------
- `Autoencoder` -- a single hidden layer encoder/decoder. The default
  choice for small-to-medium tabular data; cheapest to train.
- `DeepAutoencoder` -- a configurable multi-layer encoder/decoder
  (`hidden_dims=[...]`, geometric defaults between the input and latent
  dimensionality when not given). Use when a shallow network underfits.
- `SparseAutoencoder` -- adds an L1 penalty (`sparsity_weight=`) on the
  hidden-layer activations, encouraging a sparse (more interpretable)
  intermediate representation.
- `ConvolutionalAutoencoder` -- 1-D convolutions slide across the feature
  axis, exploiting local structure among adjacent columns (e.g. spatially
  or temporally adjacent features/channels) rather than treating every
  feature independently as a plain `Linear` layer would.
- `SequenceAutoencoder` -- a GRU seq2seq model that treats the ROWS of `x`
  as one time-ordered sequence and produces a per-timepoint latent code
  (`transform(x)` still returns `(n_rows, n_components)`, one latent
  vector per row/timepoint). Use for genuine timeseries/trajectory data
  where row order matters.
- `VariationalAutoencoder` -- a probabilistic encoder (mean + log-variance
  heads) trained with a KL-divergence term toward a standard normal prior
  via the reparameterization trick; `transform` returns the latent MEANS
  (not samples), giving a smooth, generative latent space.

All six share `_fit_torch_model` (Adam optimizer, MSE reconstruction loss
plus a variant-specific extra term) for their training loop, and the
`_TorchAutoencoderBase` mixin for the shared scikit-learn-style contract:
`fit`/`transform`/`fit_transform`/`inverse_transform` (the decoder) plus an
`n_components` attribute (wired to `hypertools.reduce.reduce.reduce`'s
`ndims=`). Inputs are standardized internally at fit time (mean/std) and
the standardization is undone in `inverse_transform`, so training is
stable even on unscaled data.
"""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from .._shared.lazy_import import lazy_import

torch = lazy_import('torch', purpose='the autoencoder reducers')   # installs [torch] on demand
nn = torch.nn
F = torch.nn.functional


def _resolve_device(device):
    """Resolve `device='auto'` to `'cuda'` > `'mps'` > `'cpu'` (in that
    order of preference); any other value is passed through to
    `torch.device` as-is.

    Parameters
    ----------
    device : str
        `'auto'`, or any string accepted by `torch.device`.

    Returns
    -------
    torch.device
        The resolved device.
    """
    if device != 'auto':
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device('cuda')
    mps = getattr(torch.backends, 'mps', None)
    if mps is not None and mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def _geometric_hidden_dims(n_in, n_components, n_layers):
    """A geometric progression of `n_layers` hidden-layer widths strictly
    between `n_in` and `n_components`, used as the default `hidden_dims`
    for the multi-layer variants (`DeepAutoencoder`, `VariationalAutoencoder`)
    when the caller does not supply one explicitly.

    Parameters
    ----------
    n_in : int
        Number of input features.
    n_components : int
        Latent dimensionality.
    n_layers : int
        Number of hidden layers to generate.

    Returns
    -------
    list of int
        `n_layers` widths, decreasing from close to `n_in` down to just
        above `n_components`.
    """
    if n_layers <= 0:
        return []
    lo = max(int(n_components) + 1, 2)
    hi = max(int(n_in), lo + 1)
    if n_layers == 1:
        return [max(int(round(np.sqrt(hi * lo))), lo)]
    sizes = np.geomspace(hi, lo, num=n_layers + 2)[1:-1]
    return [max(int(round(s)), lo) for s in sizes]


def _resolve_hidden_dims(hidden_dims, n_in, n_components, default_layers):
    """Normalize the shared `hidden_dims=` constructor kwarg to a list of
    layer widths.

    Parameters
    ----------
    hidden_dims : None, int, or sequence of int
        `None` computes `default_layers` geometric widths (see
        `_geometric_hidden_dims`); an `int` is treated as a single hidden
        layer of that width; a sequence is used exactly as given (letting
        `DeepAutoencoder`/`VariationalAutoencoder` callers control depth).
    n_in : int
        Number of input features.
    n_components : int
        Latent dimensionality.
    default_layers : int
        Number of layers to generate when `hidden_dims` is `None`.

    Returns
    -------
    list of int
        Hidden layer widths, outermost (closest to the input) first.
    """
    if hidden_dims is None:
        return _geometric_hidden_dims(n_in, n_components, default_layers)
    if isinstance(hidden_dims, (int, np.integer)):
        return [int(hidden_dims)]
    return [int(d) for d in hidden_dims]


def _fit_torch_model(net, Xs, *, epochs, batch_size, lr, device, random_state,
                      verbose, sequence_mode):
    """Shared training loop for all six autoencoder variants: Adam
    optimizer, minimizing `net.compute_loss` (each variant's `nn.Module`
    implements this -- MSE reconstruction, plus an L1 activation penalty
    for `SparseAutoencoder` or a KL term for `VariationalAutoencoder`).

    Parameters
    ----------
    net : torch.nn.Module
        A freshly constructed (untrained) autoencoder network implementing
        `encode`, `decode`, and `compute_loss`.
    Xs : numpy.ndarray
        Already-standardized (mean 0, std 1 per feature) training data.
    epochs : int
        Number of passes over the data (`epochs=0` returns `net`
        untouched, at its random initialization -- used by the test suite
        as an "untrained" baseline).
    batch_size : int
        Minibatch size (ignored when `sequence_mode=True`: the whole
        sequence is used as one batch every epoch, since shuffling rows
        would destroy the sequence's temporal order).
    lr : float
        Adam learning rate.
    device : str
        `'auto'`, or any string accepted by `torch.device` (see
        `_resolve_device`).
    random_state : int or None
        When given, seeds `torch.manual_seed` before training so
        minibatch order (and any stochastic layers, e.g. the VAE's
        reparameterization noise) is reproducible.
    verbose : bool
        Whether to print a progress line every ~10% of `epochs`.
    sequence_mode : bool
        `True` for `SequenceAutoencoder`: trains on the full sequence
        (no shuffling/minibatching) every epoch.

    Returns
    -------
    (torch.nn.Module, torch.device)
        The trained (or, for `epochs=0`, untouched) network in `eval`
        mode, and the device it was moved to.
    """
    dev = _resolve_device(device)
    net = net.to(dev)
    if random_state is not None:
        torch.manual_seed(random_state)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    x_t = torch.as_tensor(Xs, dtype=torch.float32, device=dev)
    n = x_t.shape[0]

    net.train()
    for epoch in range(epochs):
        if sequence_mode:
            optimizer.zero_grad()
            loss = net.compute_loss(x_t)
            loss.backward()
            optimizer.step()
            epoch_loss = loss.item()
        else:
            if random_state is not None:
                gen = torch.Generator().manual_seed(random_state * 1_000_003 + epoch)
                perm = torch.randperm(n, generator=gen)
            else:
                perm = torch.randperm(n)
            total = 0.0
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                batch = x_t[idx]
                optimizer.zero_grad()
                loss = net.compute_loss(batch)
                loss.backward()
                optimizer.step()
                total += loss.item() * idx.numel()
            epoch_loss = total / n
        if verbose and (epoch == 0 or epoch == epochs - 1
                        or (epoch + 1) % max(1, epochs // 10) == 0):
            print(f'[hypertools autoencoder] epoch {epoch + 1}/{epochs} '
                  f'loss={epoch_loss:.6f}')
    net.eval()
    return net, dev


# --------------------------------------------------------------------- #
# nn.Module network definitions (one per variant)
# --------------------------------------------------------------------- #

class _ShallowAENet(nn.Module):
    """Single-hidden-layer encoder/decoder, used by `Autoencoder`."""

    def __init__(self, n_in, n_components, hidden_dim):
        super().__init__()
        self.enc1 = nn.Linear(n_in, hidden_dim)
        self.enc2 = nn.Linear(hidden_dim, n_components)
        self.dec1 = nn.Linear(n_components, hidden_dim)
        self.dec2 = nn.Linear(hidden_dim, n_in)
        self.act = nn.ReLU()

    def encode(self, x):
        """Map input `x` to its latent (`n_components`-dim) representation."""
        return self.enc2(self.act(self.enc1(x)))

    def decode(self, z):
        """Reconstruct the input space from latent code `z`."""
        return self.dec2(self.act(self.dec1(z)))

    def forward(self, x):
        """Full autoencoder pass: encode `x` then decode back to the input space."""
        return self.decode(self.encode(x))

    def compute_loss(self, x):
        """MSE reconstruction loss between `x` and its autoencoded reconstruction."""
        return F.mse_loss(self.forward(x), x)


class _SparseAENet(nn.Module):
    """Single-hidden-layer encoder/decoder with an L1 penalty on the
    hidden-layer activation, used by `SparseAutoencoder`."""

    def __init__(self, n_in, n_components, hidden_dim, sparsity_weight):
        super().__init__()
        self.enc1 = nn.Linear(n_in, hidden_dim)
        self.enc2 = nn.Linear(hidden_dim, n_components)
        self.dec1 = nn.Linear(n_components, hidden_dim)
        self.dec2 = nn.Linear(hidden_dim, n_in)
        self.act = nn.ReLU()
        self.sparsity_weight = sparsity_weight

    def encode(self, x):
        """Map input `x` to its latent (`n_components`-dim) representation."""
        return self.enc2(self.act(self.enc1(x)))

    def decode(self, z):
        """Reconstruct the input space from latent code `z`."""
        return self.dec2(self.act(self.dec1(z)))

    def forward(self, x):
        """Full autoencoder pass: encode `x` then decode back to the input space."""
        return self.decode(self.encode(x))

    def compute_loss(self, x):
        """MSE reconstruction loss plus an L1 sparsity penalty on the hidden-layer activation."""
        hidden = self.act(self.enc1(x))
        z = self.enc2(hidden)
        xhat = self.dec2(self.act(self.dec1(z)))
        recon = F.mse_loss(xhat, x)
        l1 = hidden.abs().mean()
        return recon + self.sparsity_weight * l1


class _DeepAENet(nn.Module):
    """Multi-layer (`hidden_dims`) encoder/decoder, used by
    `DeepAutoencoder`."""

    def __init__(self, n_in, n_components, hidden_dims):
        super().__init__()
        dims = [n_in] + list(hidden_dims)
        enc_layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            enc_layers += [nn.Linear(a, b), nn.ReLU()]
        enc_layers.append(nn.Linear(dims[-1], n_components))
        self.encoder = nn.Sequential(*enc_layers)

        rdims = [n_components] + list(reversed(hidden_dims))
        dec_layers = []
        for a, b in zip(rdims[:-1], rdims[1:]):
            dec_layers += [nn.Linear(a, b), nn.ReLU()]
        dec_layers.append(nn.Linear(rdims[-1], n_in))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x):
        """Map input `x` to its latent (`n_components`-dim) representation via the multi-layer encoder."""
        return self.encoder(x)

    def decode(self, z):
        """Reconstruct the input space from latent code `z` via the multi-layer decoder."""
        return self.decoder(z)

    def forward(self, x):
        """Full autoencoder pass: encode `x` then decode back to the input space."""
        return self.decode(self.encode(x))

    def compute_loss(self, x):
        """MSE reconstruction loss between `x` and its autoencoded reconstruction."""
        return F.mse_loss(self.forward(x), x)


class _ConvAENet(nn.Module):
    """1-D convolutional encoder/decoder sliding over the feature axis,
    used by `ConvolutionalAutoencoder`."""

    def __init__(self, n_in, n_components, channels):
        super().__init__()
        c1, c2 = channels
        self.enc_conv = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(c1, c2, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.enc_fc = nn.Linear(c2 * n_in, n_components)
        self.dec_fc = nn.Linear(n_components, c2 * n_in)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose1d(c2, c1, kernel_size=3, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(c1, 1, kernel_size=3, padding=1),
        )
        self.n_in = n_in
        self.c2 = c2

    def encode(self, x):
        """Map input `x` to its latent (`n_components`-dim) representation via 1-D convolutions."""
        h = self.enc_conv(x.unsqueeze(1))
        return self.enc_fc(h.flatten(1))

    def decode(self, z):
        """Reconstruct the input space from latent code `z` via transposed 1-D convolutions."""
        h = self.dec_fc(z).view(-1, self.c2, self.n_in)
        return self.dec_conv(h).squeeze(1)

    def forward(self, x):
        """Full autoencoder pass: encode `x` then decode back to the input space."""
        return self.decode(self.encode(x))

    def compute_loss(self, x):
        """MSE reconstruction loss between `x` and its autoencoded reconstruction."""
        return F.mse_loss(self.forward(x), x)


class _SeqAENet(nn.Module):
    """GRU seq2seq encoder/decoder, used by `SequenceAutoencoder`. Treats
    its whole input as ONE time-ordered sequence (`(seq_len, n_in)`) and
    produces one latent vector per timepoint (`(seq_len, n_components)`)."""

    def __init__(self, n_in, n_components, hidden_dim):
        super().__init__()
        self.encoder_rnn = nn.GRU(n_in, hidden_dim, batch_first=True)
        self.enc_fc = nn.Linear(hidden_dim, n_components)
        self.decoder_rnn = nn.GRU(n_components, hidden_dim, batch_first=True)
        self.dec_fc = nn.Linear(hidden_dim, n_in)

    def encode(self, x):
        """Map the sequence `x` (`(seq_len, n_in)`) to one latent vector per timepoint via a GRU encoder."""
        h, _ = self.encoder_rnn(x.unsqueeze(0))
        return self.enc_fc(h.squeeze(0))

    def decode(self, z):
        """Reconstruct the sequence's input space from its per-timepoint latent codes `z` via a GRU decoder."""
        h, _ = self.decoder_rnn(z.unsqueeze(0))
        return self.dec_fc(h.squeeze(0))

    def forward(self, x):
        """Full autoencoder pass: encode sequence `x` then decode back to the input space."""
        return self.decode(self.encode(x))

    def compute_loss(self, x):
        """MSE reconstruction loss between `x` and its autoencoded reconstruction."""
        return F.mse_loss(self.forward(x), x)


class _VAENet(nn.Module):
    """Variational encoder (mean/log-variance heads) + decoder, used by
    `VariationalAutoencoder`."""

    def __init__(self, n_in, n_components, hidden_dims, kl_weight):
        super().__init__()
        dims = [n_in] + list(hidden_dims)
        enc_layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            enc_layers += [nn.Linear(a, b), nn.ReLU()]
        self.enc_body = nn.Sequential(*enc_layers)
        self.mu = nn.Linear(dims[-1], n_components)
        self.logvar = nn.Linear(dims[-1], n_components)

        rdims = [n_components] + list(reversed(hidden_dims))
        dec_layers = []
        for a, b in zip(rdims[:-1], rdims[1:]):
            dec_layers += [nn.Linear(a, b), nn.ReLU()]
        dec_layers.append(nn.Linear(rdims[-1], n_in))
        self.decoder = nn.Sequential(*dec_layers)
        self.kl_weight = kl_weight

    def encode_params(self, x):
        """Map input `x` to the latent Gaussian's parameters `(mu, logvar)`."""
        h = self.enc_body(x)
        return self.mu(h), self.logvar(h)

    def encode(self, x):
        """Map input `x` to its latent MEAN (`mu`), not a stochastic sample (used by `transform`)."""
        # transform() returns the latent MEANS, not stochastic samples.
        mu, _ = self.encode_params(x)
        return mu

    def reparameterize(self, mu, logvar):
        """Sample a latent code from `N(mu, exp(logvar))` via the reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Reconstruct the input space from latent code `z`."""
        return self.decoder(z)

    def forward(self, x):
        """Full VAE pass: encode `x` to `(mu, logvar)`, sample `z`, and decode `z`.

        Returns
        -------
        tuple of (Tensor, Tensor, Tensor)
            `(reconstruction, mu, logvar)`.
        """
        mu, logvar = self.encode_params(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def compute_loss(self, x):
        """MSE reconstruction loss plus a KL-divergence term toward a standard normal prior."""
        xhat, mu, logvar = self.forward(x)
        recon = F.mse_loss(xhat, x)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + self.kl_weight * kl


# --------------------------------------------------------------------- #
# scikit-learn-style wrapper classes
# --------------------------------------------------------------------- #

class _TorchAutoencoderBase(BaseEstimator):
    """Shared scikit-learn-style contract for the six torch autoencoder
    reducers: `fit`/`transform`/`fit_transform`/`inverse_transform`
    (the decoder), standardizing inputs at fit time and undoing that
    standardization in `inverse_transform`.

    Subclasses implement `_build_net(n_features_in)`, returning a freshly
    constructed (untrained) `torch.nn.Module` exposing `encode`, `decode`,
    and `compute_loss` (see the `_*Net` classes above), and set
    `_sequence_mode = True` if training should not shuffle/minibatch rows
    (only `SequenceAutoencoder` does this).

    Attributes
    ----------
    net_ : torch.nn.Module or None
        The fitted network (`None` until `fit`/`fit_transform` runs).
    mean_, std_ : numpy.ndarray or None
        Per-feature fit-time mean/std used to standardize inputs.
    n_features_in_ : int or None
        Number of input features, set by `fit`/`fit_transform`.
    """

    _sequence_mode = False

    def _build_net(self, n_features_in):
        raise NotImplementedError

    @property
    def is_fitted(self):
        """Whether `fit`/`fit_transform` has already been run."""
        return self.net_ is not None

    def fit(self, X, y=None):
        """Fit the autoencoder on `X` (see `fit_transform`); returns
        `self`."""
        self.fit_transform(X)
        return self

    def fit_transform(self, X, y=None):
        """Standardize `X`, train a fresh network on it, and return the
        fitted latent codes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        numpy.ndarray of shape (n_samples, n_components)
            The latent codes (for `VariationalAutoencoder`, the latent
            means).

        Raises
        ------
        ValueError
            If `X` is not 2-D, or if a training hyperparameter is invalid
            (`epochs` must be a non-negative integer, `batch_size` a
            positive integer, and `lr` a positive finite number).
        """
        # validate training hyperparameters up front: a typo'd `epochs=-5`
        # used to run "successfully" and return a finite but UNTRAINED
        # embedding with no error or warning (release-1.0 audit,
        # D04-gallery-models-010).
        if isinstance(self.epochs, bool) \
                or not isinstance(self.epochs, (int, np.integer)) \
                or self.epochs < 0:
            raise ValueError(
                f'epochs must be a non-negative integer (0 explicitly '
                f'skips training, leaving the network at its random '
                f'initialization); got {self.epochs!r}')
        if isinstance(self.batch_size, bool) \
                or not isinstance(self.batch_size, (int, np.integer)) \
                or self.batch_size < 1:
            raise ValueError(
                f'batch_size must be a positive integer; got '
                f'{self.batch_size!r}')
        if isinstance(self.lr, bool) \
                or not isinstance(self.lr, (int, float, np.integer,
                                            np.floating)) \
                or not np.isfinite(self.lr) or self.lr <= 0:
            raise ValueError(
                f'lr (learning rate) must be a positive finite number; '
                f'got {self.lr!r}')
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError('X must be a 2D array')
        self.n_features_in_ = X.shape[1]
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0
        self.std_ = std
        xs = (X - self.mean_) / self.std_

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        net = self._build_net(self.n_features_in_)
        net, device = _fit_torch_model(
            net, xs, epochs=self.epochs, batch_size=self.batch_size,
            lr=self.lr, device=self.device, random_state=self.random_state,
            verbose=self.verbose, sequence_mode=self._sequence_mode)
        self.net_ = net
        self.device_ = device
        return self._encode(xs)

    def transform(self, X):
        """Apply the already-fitted network to (new) `X`, without
        refitting.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to encode.

        Returns
        -------
        numpy.ndarray of shape (n_samples, n_components)
            The latent codes.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If `fit`/`fit_transform` has not been called yet.
        """
        if self.net_ is None:
            raise NotFittedError('must fit autoencoder before transforming data')
        X = np.asarray(X, dtype=np.float64)
        xs = (X - self.mean_) / self.std_
        return self._encode(xs)

    def _encode(self, xs):
        self.net_.eval()
        with torch.no_grad():
            x_t = torch.as_tensor(xs, dtype=torch.float32, device=self.device_)
            z = self.net_.encode(x_t)
            return z.detach().cpu().numpy().astype(np.float64)

    def inverse_transform(self, Z):
        """Decode latent codes `Z` back to (standardization-undone)
        feature space.

        Parameters
        ----------
        Z : array-like of shape (n_samples, n_components)
            Latent codes (e.g. from `transform`).

        Returns
        -------
        numpy.ndarray of shape (n_samples, n_features)
            The reconstructed data.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If `fit`/`fit_transform` has not been called yet.
        """
        if self.net_ is None:
            raise NotFittedError('must fit autoencoder before inverse-transforming data')
        self.net_.eval()
        with torch.no_grad():
            z_t = torch.as_tensor(np.asarray(Z, dtype=np.float64),
                                   dtype=torch.float32, device=self.device_)
            xhat = self.net_.decode(z_t).detach().cpu().numpy().astype(np.float64)
        return xhat * self.std_ + self.mean_


class Autoencoder(_TorchAutoencoderBase):
    """A single-hidden-layer (shallow) autoencoder reducer.

    The default, cheapest-to-train autoencoder variant: one `Linear` +
    ReLU hidden layer on each side of the bottleneck. Good for small to
    medium tabular data where a deeper network is unnecessary.

    Parameters
    ----------
    n_components : int
        Latent (bottleneck) dimensionality (default: 2). Wired to
        `hypertools.reduce.reduce.reduce`'s `ndims=`.
    epochs : int
        Number of training epochs (default: 100). Must be a non-negative
        integer (validated at fit time); `epochs=0` explicitly skips
        training, leaving the network at its random initialization
        (useful only as an untrained baseline).
    batch_size : int
        Minibatch size (default: 64). Must be a positive integer
        (validated at fit time).
    lr : float
        Adam learning rate (default: 1e-3). Must be a positive finite
        number (validated at fit time).
    hidden_dims : None, int, or sequence of int
        Hidden layer width. `None` (default) computes a sensible width
        geometrically between `n_components` and the number of input
        features; an `int` (or single-element sequence) sets it directly.
    device : str
        `'auto'` (default) picks `'cuda'` > `'mps'` > `'cpu'`; any other
        string is passed to `torch.device` as-is.
    random_state : int or None
        Seeds `torch.manual_seed` for reproducible weight initialization
        and minibatch order (default: None, i.e. nondeterministic).
    verbose : bool
        Print training progress (default: False).

    Attributes
    ----------
    net_ : torch.nn.Module or None
        The fitted network.
    """

    def __init__(self, n_components=2, epochs=100, batch_size=64, lr=1e-3,
                 hidden_dims=None, device='auto', random_state=None,
                 verbose=False):
        self.n_components = n_components
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.hidden_dims = hidden_dims
        self.device = device
        self.random_state = random_state
        self.verbose = verbose
        self.net_ = None
        self.mean_ = None
        self.std_ = None
        self.n_features_in_ = None
        self.device_ = None

    def _build_net(self, n_features_in):
        hidden = _resolve_hidden_dims(self.hidden_dims, n_features_in,
                                       self.n_components, default_layers=1)[0]
        return _ShallowAENet(n_features_in, self.n_components, hidden)


class SparseAutoencoder(_TorchAutoencoderBase):
    """A single-hidden-layer autoencoder with an L1 penalty on the hidden
    activation (sparse coding), encouraging a sparser, more interpretable
    intermediate representation.

    Parameters
    ----------
    n_components : int
        Latent (bottleneck) dimensionality (default: 2).
    sparsity_weight : float
        Weight of the L1 penalty on the mean absolute hidden-layer
        activation, added to the MSE reconstruction loss (default: 1e-3).
    epochs, batch_size, lr, hidden_dims, device, random_state, verbose
        See `Autoencoder`.

    Attributes
    ----------
    net_ : torch.nn.Module or None
        The fitted network.
    """

    def __init__(self, n_components=2, sparsity_weight=1e-3, epochs=100,
                 batch_size=64, lr=1e-3, hidden_dims=None, device='auto',
                 random_state=None, verbose=False):
        self.n_components = n_components
        self.sparsity_weight = sparsity_weight
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.hidden_dims = hidden_dims
        self.device = device
        self.random_state = random_state
        self.verbose = verbose
        self.net_ = None
        self.mean_ = None
        self.std_ = None
        self.n_features_in_ = None
        self.device_ = None

    def _build_net(self, n_features_in):
        hidden = _resolve_hidden_dims(self.hidden_dims, n_features_in,
                                       self.n_components, default_layers=1)[0]
        return _SparseAENet(n_features_in, self.n_components, hidden,
                             self.sparsity_weight)


class DeepAutoencoder(_TorchAutoencoderBase):
    """A multi-layer autoencoder reducer with configurable depth.

    Parameters
    ----------
    n_components : int
        Latent (bottleneck) dimensionality (default: 2).
    hidden_dims : None, int, or sequence of int
        Hidden layer widths, outermost (closest to the input) first.
        `None` (default) computes 2 geometrically-spaced widths between
        the number of input features and `n_components`; a sequence sets
        the exact depth and widths (e.g. `[64, 32, 16]`).
    epochs, batch_size, lr, device, random_state, verbose
        See `Autoencoder`.

    Attributes
    ----------
    net_ : torch.nn.Module or None
        The fitted network.
    """

    def __init__(self, n_components=2, hidden_dims=None, epochs=100,
                 batch_size=64, lr=1e-3, device='auto', random_state=None,
                 verbose=False):
        self.n_components = n_components
        self.hidden_dims = hidden_dims
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.random_state = random_state
        self.verbose = verbose
        self.net_ = None
        self.mean_ = None
        self.std_ = None
        self.n_features_in_ = None
        self.device_ = None

    def _build_net(self, n_features_in):
        hidden_dims = _resolve_hidden_dims(self.hidden_dims, n_features_in,
                                            self.n_components, default_layers=2)
        return _DeepAENet(n_features_in, self.n_components, hidden_dims)


class ConvolutionalAutoencoder(_TorchAutoencoderBase):
    """A 1-D convolutional autoencoder reducer: two `Conv1d` layers slide
    across the feature axis (treating each row as a length-`n_features`,
    1-channel signal), exploiting local structure among adjacent features
    rather than treating every feature independently.

    Parameters
    ----------
    n_components : int
        Latent (bottleneck) dimensionality (default: 2).
    hidden_dims : None or sequence of 2 int
        `(channels_1, channels_2)` for the two `Conv1d`/`ConvTranspose1d`
        layers. `None` (default) uses `(8, 16)`.
    epochs, batch_size, lr, device, random_state, verbose
        See `Autoencoder`.

    Attributes
    ----------
    net_ : torch.nn.Module or None
        The fitted network.
    """

    def __init__(self, n_components=2, hidden_dims=None, epochs=100,
                 batch_size=64, lr=1e-3, device='auto', random_state=None,
                 verbose=False):
        self.n_components = n_components
        self.hidden_dims = hidden_dims
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.random_state = random_state
        self.verbose = verbose
        self.net_ = None
        self.mean_ = None
        self.std_ = None
        self.n_features_in_ = None
        self.device_ = None

    def _build_net(self, n_features_in):
        if self.hidden_dims is None:
            channels = (8, 16)
        elif isinstance(self.hidden_dims, (int, np.integer)):
            channels = (int(self.hidden_dims), int(self.hidden_dims))
        else:
            dims = [int(d) for d in self.hidden_dims]
            channels = (dims[0], dims[1] if len(dims) > 1 else dims[0])
        return _ConvAENet(n_features_in, self.n_components, channels)


class SequenceAutoencoder(_TorchAutoencoderBase):
    """A GRU seq2seq autoencoder reducer for genuine timeseries/trajectory
    data: the ROWS of `x` are treated as one time-ordered sequence, and a
    latent vector is produced PER TIMEPOINT (`transform(x)` still returns
    `(n_rows, n_components)`, one row per input row).

    Because row order is meaningful, training uses the full sequence every
    epoch (no row shuffling/minibatching -- `batch_size` is accepted for
    interface consistency with the other variants but ignored).

    Parameters
    ----------
    n_components : int
        Per-timepoint latent dimensionality (default: 2).
    hidden_dims : None, int, or sequence of int
        GRU hidden size. `None` (default) computes a sensible width
        geometrically between `n_components` and the number of input
        features; only the first value is used if a sequence is given.
    epochs, lr, device, random_state, verbose
        See `Autoencoder`.
    batch_size : int
        Accepted for interface consistency; ignored (see above).

    Attributes
    ----------
    net_ : torch.nn.Module or None
        The fitted network.
    """

    _sequence_mode = True

    def __init__(self, n_components=2, hidden_dims=None, epochs=100,
                 batch_size=64, lr=1e-3, device='auto', random_state=None,
                 verbose=False):
        self.n_components = n_components
        self.hidden_dims = hidden_dims
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.random_state = random_state
        self.verbose = verbose
        self.net_ = None
        self.mean_ = None
        self.std_ = None
        self.n_features_in_ = None
        self.device_ = None

    def _build_net(self, n_features_in):
        hidden = _resolve_hidden_dims(self.hidden_dims, n_features_in,
                                       self.n_components, default_layers=1)[0]
        return _SeqAENet(n_features_in, self.n_components, hidden)


class VariationalAutoencoder(_TorchAutoencoderBase):
    """A variational autoencoder (VAE) reducer: a probabilistic encoder
    (mean + log-variance heads) trained with a KL-divergence term toward a
    standard normal prior, via the reparameterization trick.
    `transform`/`fit_transform` return the latent MEANS (not stochastic
    samples), giving a smooth, reproducible, generative latent space.

    Parameters
    ----------
    n_components : int
        Latent dimensionality (default: 2).
    kl_weight : float
        Weight of the KL-divergence term, added to the MSE reconstruction
        loss (default: 1.0).
    hidden_dims : None, int, or sequence of int
        Hidden layer widths of the shared encoder body / decoder,
        outermost (closest to the input) first. `None` (default) computes
        2 geometrically-spaced widths.
    epochs, batch_size, lr, device, random_state, verbose
        See `Autoencoder`.

    Attributes
    ----------
    net_ : torch.nn.Module or None
        The fitted network.
    """

    def __init__(self, n_components=2, kl_weight=1.0, hidden_dims=None,
                 epochs=100, batch_size=64, lr=1e-3, device='auto',
                 random_state=None, verbose=False):
        self.n_components = n_components
        self.kl_weight = kl_weight
        self.hidden_dims = hidden_dims
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.random_state = random_state
        self.verbose = verbose
        self.net_ = None
        self.mean_ = None
        self.std_ = None
        self.n_features_in_ = None
        self.device_ = None

    def _build_net(self, n_features_in):
        hidden_dims = _resolve_hidden_dims(self.hidden_dims, n_features_in,
                                            self.n_components, default_layers=2)
        return _VAENet(n_features_in, self.n_components, hidden_dims,
                        self.kl_weight)


#: names resolvable by `hypertools.reduce.common.resolve_reducer`, lazily
#: importing this module (torch) only when one is actually requested.
AUTOENCODER_NAMES = (
    'Autoencoder', 'DeepAutoencoder', 'SparseAutoencoder',
    'ConvolutionalAutoencoder', 'SequenceAutoencoder',
    'VariationalAutoencoder',
)
