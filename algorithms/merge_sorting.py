from collections import deque
    
def merge(test1, test2):
  merged = []
  test1, test2 = deque(test1), deque(test2)
  while test1:
      if test2:
          merged.append(test1.popleft()) if test1[0] <= test2[0] else merged.append(test2.popleft())
      else: merged.append(test1.popleft())
  merged.extend(test2)
  return merged
  
def solve(dlist):
  dlen = len(dlist)
  if dlen == 1: return dlist
  elif dlen == 2: return dlist if dlist[0] <= dlist[1] else [dlist[1], dlist[0]]
  else:
    pivot = dlen // 2
    return merge(solve(dlist[:pivot]), solve(dlist[pivot:]))