from numpy.core.fromnumeric import prod, product


def NAND(input):
    return 1-product(input)

def AND(input):
    return product(input)

def OR(input):
    return max(input)

def NOR(input):
    return 1-max(input)

def eightBitAdd(input):
    return sum(input) % 256

def eightBitMultiply(input):
    return product(input) % 256

#### dispatcher for config files to be able to find functions
operations_dispatcher = {   "NAND" : NAND,
                            "AND" : AND,
                            "OR" : OR,
                            "NOR" : NOR,
                            "eightBitAdd" : eightBitAdd,
                            "eightBitMultiply" : eightBitMultiply,
                            "sum" : sum,
                            "product" : product,
                            "min" : min,
                            "max" : max}