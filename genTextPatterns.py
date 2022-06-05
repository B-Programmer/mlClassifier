# Define a function to perform this operation
def genTextPatterns(text, i, c):
    textList = []
    # Convert the text to a list
    textList += text
    # Remove the ith element/character of the list
    a = textList.pop(i)
    # Add the removed into the middle/center of the list
    textList.insert(c, a)
    return textList