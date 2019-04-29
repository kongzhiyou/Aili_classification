def test():
    i = 3
    while True:
        i += 1
        res = yield i
        print(res)

def value_print(i):
    print(i)

test()
