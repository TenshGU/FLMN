from queue import Queue

q = Queue(maxsize=5)
q.put(1)
print(q.get())
