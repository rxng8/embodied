from . import printing


def map_(fn, *trees, isleaf=None):
  assert trees, 'Provide one or more nested Python structures'
  kw = dict(isleaf=isleaf)
  first = trees[0]
  assert all(isinstance(x, type(first)) for x in trees)
  if isleaf and isleaf(first):
    return fn(*trees)
  if isinstance(first, list):
    assert all(len(x) == len(first) for x in trees), printing.format_(trees)
    return [map_(
        fn, *[t[i] for t in trees], **kw) for i in range(len(first))]
  if isinstance(first, tuple):
    assert all(len(x) == len(first) for x in trees), printing.format_(trees)
    return tuple([map_(
        fn, *[t[i] for t in trees], **kw) for i in range(len(first))])
  if isinstance(first, dict):
    assert all(set(x.keys()) == set(first.keys()) for x in trees), (
        printing.format_(trees))
    return {k: map_(fn, *[t[k] for t in trees], **kw) for k in first}
  if hasattr(first, 'keys') and hasattr(first, 'get'):
    assert all(set(x.keys()) == set(first.keys()) for x in trees), (
        printing.format_(trees))
    return type(first)(
        {k: map_(fn, *[t[k] for t in trees], **kw) for k in first})
  return fn(*trees)

def leaves_(tree, is_leaf=None):
  kw = dict(is_leaf=is_leaf)
  result = []
  if is_leaf and is_leaf(tree):
    result.append(tree)
  if isinstance(tree, list):
    for t in tree:
      li = leaves_(t, **kw)
      [result.append(item) for item in li]
  elif isinstance(tree, tuple):
   for t in tree:
      li = leaves_(t, **kw)
      [result.append(item) for item in li]
  elif isinstance(tree, dict):
    for k, v in tree.items():
      li = leaves_(v, **kw)
      [result.append(item) for item in li]
  elif hasattr(tree, 'keys') and hasattr(tree, 'get'):
    for k in tree:
      li = leaves_(tree[k], **kw)
      [result.append(item) for item in li]
  else:
    result.append(tree)
  return result

leaves = leaves_
map = map_
