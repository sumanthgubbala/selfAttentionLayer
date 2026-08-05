import numpy as np

vocabs={
    "i": 0,
    "love": 1,
    "hate": 2,
    "this": 3,
    "movie": 4,
    "positive": 5,
    "negative": 6,
    "worst": 7,
    "good": 8,
    "like": 9,


}


embeddings = np.array([
    [1,0], # i
    [1,2], # love
    [-1,2], # hate
    [0,1], # this
    [0,2] , # movie,
    [1,0], # positive
    [0,0], # negative
    [-1,1], # worst
    [1,1],  # good
    [1,3]   # like


])


sentence = ["i", "like", "this","movie"]

x = np.array([embeddings[vocabs[word]] for word in sentence])

print(x)


Q = x
k = x
v = x

scores = Q @ k.T
print(scores)

def softmax(x):
    exp = np.exp(x-np.max(x, axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)

attention = softmax(scores)
print(attention)


output = attention @ v
print(output)

sentence_output = output.mean(axis=0)
print(sentence_output)

if sentence_output[0] > 0.5:
    print("Positive sentiment")
elif sentence_output[0]>0:
    print("Neutral")
else:
    print("Negative sentiment")