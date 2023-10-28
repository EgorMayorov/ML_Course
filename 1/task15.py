from typing import List


def hello(name=None) -> str:
    if (name is None) | (name == ""):
        return "Hello!"
    else:
        return "Hello, " + name + "!"


def int_to_roman(num: int) -> str:
    n = num
    r = (n // 1000) * "M"
    n = n % 1000
    r = r + (n // 900) * "CM"
    n = n % 900
    r = r + (n // 500) * "D"
    n = n % 500
    r += (n // 400) * "CD"
    n = n % 400
    r += (n // 100) * "C"
    n = n % 100
    r = r + (n // 90) * "XC"
    n = n % 90
    r = r + (n // 50) * "L"
    n = n % 50
    r += (n // 40) * "XL"
    n = n % 40
    r += (n // 10) * "X"
    n = n % 10
    r = r + (n // 9) * "IX"
    n = n % 9
    r = r + (n // 5) * "V"
    n = n % 5
    r += (n // 4) * "IV"
    n = n % 4
    r += (n // 1) * "I"
    return r


def longest_common_prefix(strs_input: List[str]) -> str:
    if strs_input == []:
        return ''
    lst = List.copy(strs_input)
    for s in range(len(lst)):
        lst[s] = lst[s].lstrip()
    minstr = min(lst, key=len)
    ln = len(minstr)
    for j in range(0, ln):
        b = True
        for i in lst:
            if minstr not in i:
                b = False
                break
        if b:
            return minstr
        minstr = minstr[:-1]
    return ''


def primes() -> int:
    k = 1
    while True:
        k += 1
        b = True
        for i in range(2, k // 2 + 1):
            if k % i == 0:
                b = False
                break
        if b:
            yield k


class BankCard:
    def __init__(self, total_sum: int, balance_limit=0):
        self.total_sum = total_sum
        self.balance_limit = balance_limit

    def __call__(self, item):
        if item > self.total_sum:
            print("Not enough money to spend", item, "dollars.")
            raise TypeError
        self.total_sum -= item
        print("You spent", item, "dollars.")

    def put(self, sum_put):
        self.total_sum += sum_put
        print("You put", sum_put, "dollars.")

    def balance(self):
        self.balance_limit -= 1
        if (self.balance_limit):
            print("Balance check limits exceeded.")
            raise ValueError
        return self.total_sum

    def get_balance(self):
        self.balance_limit -= 1
        return self.total_sum

    balance = property(get_balance)

    def __str__(self):
        return "To learn the balance call balance."

    def __add__(self, other):
        return BankCard(self.total_sum + other.total_sum,
                        max(self.balance_limit, other.balance_limit))
