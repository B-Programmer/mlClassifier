# Creating a function to convert binary to decimal
def binaryToDecimal(binaryList):
    decimal, n = 0, len(binaryList)
    for b in binaryList:
        decimal += b * 2 ** (n - 1)
        n -= 1
    return decimal