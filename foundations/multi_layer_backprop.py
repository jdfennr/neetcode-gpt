import numpy as np
from typing import List

class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        
        # 0. Conversions
        x = np.array(x)
        W1, W2 = np.array(W1), np.array(W2)
        b1, b2 = np.array(b1), np.array(b2)
        y_true = np.array(y_true)

        fn1 = W1 @ x + b1

        fa1 = np.maximum(0, fn1)

        fn2 = W2 @ fa1 + b2

        loss = ((fn2 - y_true) ** 2)


        dfn2 = 2 * (fn2 - y_true)

        dW2 = np.outer(dfn2, fa1)
        db2 = dfn2

        dfa1 = W2.T @ dfn2
        dfn1 = dfa1 * (fn1 > 0)

        dW1 = np.outer(dfn1, x)
        db1 = dfn1




        


        # 3. FORMAT OUTPUT
        return {
            'loss': round(loss[0], 4),
            'dW1': np.round(dW1, 4),
            'db1': np.round(db1, 4),
            'dW2': np.round(dW2, 4),
            'db2': np.round(db2, 4)
        }

