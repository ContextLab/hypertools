import json, sys
path = sys.argv[1]
texts = []
with open(path) as f:
    for line in f:
        line=line.strip()
        if not line: continue
        try: obj=json.loads(line)
        except: continue
        # find assistant messages with text content
        msg = obj.get('message') if isinstance(obj,dict) else None
        if isinstance(msg,dict) and msg.get('role')=='assistant':
            for c in msg.get('content',[]):
                if isinstance(c,dict) and c.get('type')=='text' and c.get('text','').strip():
                    texts.append(c['text'])
        # some formats: obj['type']=='assistant'
        elif isinstance(obj,dict) and obj.get('type')=='assistant' and isinstance(obj.get('content'),list):
            for c in obj['content']:
                if isinstance(c,dict) and c.get('type')=='text' and c.get('text','').strip():
                    texts.append(c['text'])
if not texts:
    print("NO ASSISTANT TEXT FOUND"); sys.exit(0)
# print the last substantial one (the final report)
final = texts[-1]
# if last is tiny (e.g. "Done."), pick the longest instead
if len(final) < 400:
    final = max(texts, key=len)
print(final)
