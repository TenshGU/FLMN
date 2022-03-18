from collections import OrderedDict


class LRUCache:
    def __init__(self, capacity=2048):
        self.dict = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key in self.dict:
            val = self.dict[key]
            self.dict.move_to_end(key)
            return val
        else:
            return -2

    def put(self, key, value):
        if key in self.dict:
            del self.dict[key]
            self.dict[key] = value
        else:
            self.dict[key] = value
            if len(self.dict) > self.capacity:
                self.dict.popitem(last=False)

    def delete(self, key):
        self.dict.pop(key)

class T:
    def __init__(self):
        self.cache = LRUCache(2)

    def get(self):
        print(self.cache.get("s"))

if __name__ == '__main__':
   cache = LRUCache(2)
   cache.put("www", 1)
   cache.put("kkk", 2)
   cache.put("sss", 3)
   print(cache.get("sss"))
   print(cache.get("www"))

   cache.delete("sss")
   print(cache.get("sss"))
