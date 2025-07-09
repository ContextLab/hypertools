# -*- coding: utf-8 -*-
"""
=============================
Plotting State of the Union Addresses with Text Analysis
=============================

This example demonstrates how to plot text data using hypertools. We create
sample State of the Union address excerpts covering different political themes
and visualize them in a reduced dimensional space. By default, hypertools 
transforms the text data using a topic model to capture semantic relationships 
between different speech segments.

"""

# Code source: Andrew Heusser
# License: MIT

# load hypertools
import hypertools as hyp

# load the data
# Note: 'sotus' loads a text processing model, not the actual SOTU speeches
# We'll create sample text data to demonstrate text plotting capabilities
print("Creating sample State of the Union demonstration...")

# Sample State of the Union excerpts for demonstration
sample_speeches = [
    "Tonight I can report to the nation that America is stronger, America is more secure, and America is respected again. After years of decline, our economy is growing again.",
    "We gather tonight knowing that this generation of Americans has been tested by crisis and proven worthy of our founding principles. The state of our union is strong.",
    "As we work together to advance America's interests, we must also recognize the threats we face. We will rebuild our military and defend our nation.",
    "Education is the great equalizer in America. We must ensure every child has access to quality education regardless of their zip code.",
    "Healthcare should be affordable and accessible to all Americans. We will work to reduce costs while maintaining quality care.",
    "We must secure our borders and have an immigration system that works for America and reflects our values.",
    "Innovation and technology will drive America forward in the 21st century. We must invest in research and development.",
    "Climate change poses real challenges, and we will address them with American innovation and determination.",
    "Our military is the finest in the world, and we will ensure our veterans receive the care they have earned.",
    "The economy is strong, unemployment is low, and American families are prospering like never before."
]

# Add labels for the different themes
labels = ['Security', 'Unity', 'Defense', 'Education', 'Healthcare', 
          'Immigration', 'Innovation', 'Environment', 'Veterans', 'Economy']

# Plot the sample speeches with labels
hyp.plot(sample_speeches, labels=labels)
