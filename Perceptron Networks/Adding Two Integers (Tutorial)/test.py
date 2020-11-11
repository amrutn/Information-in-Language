import numpy as np

operations = ['+','-','*','/']
X = []
Y = []
sentences = []
def one_hot(symbols):
    '''
    Converts symbol ('+', '=', int. between -5 to 5) array into 'stacked' one-hot vectors as specified above.
    '''
    vector_stack = []
    for symbol in symbols:
        vector = np.zeros(16)
        if symbol == '+':
            vector[0] = 1
        elif symbol == '-':
            vector[1] = 1
        elif symbol == '*':
            vector[2] = 1
        elif symbol == '/':
            vector[3] = 1
        elif symbol =='=':
            vector[4] = 1
        else:
            idx = int(symbol) + 10
            vector[idx] = 1
        vector_stack = np.concatenate((vector_stack, vector))
    return np.asarray(vector_stack)


for i in np.arange(-5, 6):
    for j in np.arange(-5, 6):
        for k in np.arange(-5, 6):
            for o in operations:
                z = [i, j, k]
                sentence = np.array([z[0], o, z[1], '=', z[2]])
                sentences.append(sentence)
                X.append(one_hot(sentence))

                # Checking for veracity
                if o is '+':
                    if z[0] + z[1] == z[2]:
                        Y.append(1)
                    else:
                        Y.append(0)
                elif o is '-':
                    if z[0] - z[1] == z[2]:
                        Y.append(1)
                    else:
                        Y.append(0)
                elif o is '*':
                    if z[0] * z[1] == z[2]:
                        Y.append(1)
                    else:
                        Y.append(0)
                elif o is '/':
                    if z[1] == 0:
                        Y.append(0)
                    elif z[0] / z[1] == z[2]:
                        Y.append(1)
                    else:
                        Y.append(0)
                else:
                    Y.append(0)


print()