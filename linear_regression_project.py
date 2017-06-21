
# coding: utf-8

# # 1 矩阵运算
# 
# ## 1.1 创建一个 4*4 的单位矩阵

# In[7]:


# 这个项目设计来帮你熟悉 python list 和线性代数
# 你不能调用任何python库，包括NumPy，来完成作业

A = [[1,2,3], 
     [2,3,3], 
     [1,2,5]]

B = [[1,2,3,5], 
     [2,3,3,5], 
     [1,2,5,1]]

# 创建一个 4*4 单位矩阵
I = [[1,0,0,0],
     [0,1,0,0],
     [0,0,1,0],
     [0,0,0,1]]


# ## 1.2 返回矩阵的行数和列数

# In[8]:


# 返回矩阵的行数和列数
def shape(M):
    try:
        row = len(M)
        col = len(M[0])
        return row, col
    except:
        return 0,0


# ## 1.3 每个元素四舍五入到特定小数数位

# In[9]:


from decimal import Decimal, ROUND_HALF_UP
# 每个元素四舍五入到特定小数数位
# 直接修改参数矩阵，无返回值
def matxRound(M, decPts=4):
    places = Decimal(10) ** -decPts
    row, col = shape(M)
    for i in range(row):
        for j in range(col):
            M[i][j] = round(M[i][j], decPts)


# ## 1.4 计算矩阵的转置

# In[10]:


# 计算矩阵的转置
def transpose(M):
    return [list(row) for row in zip(*M)]


# ## 1.5 计算矩阵乘法 AB

# In[11]:


# 计算矩阵乘法 AB，如果无法相乘则返回None
def matxMultiply(A, B):
    aRow, aCol = shape(A)
    bRow, bCol = shape(B)

    if (aCol != bRow): return None

    resultMatix = [[0]*bCol for i in range(aRow)]
    for i in range(aRow):
        for j in range(bCol):
            for k in range(bRow):
                resultMatix[i][j] += A[i][k] * B[k][j]
    return resultMatix


# ## 1.6 测试你的函数是否实现正确

# **提示：** 你可以用`from pprint import pprint`来更漂亮的打印数据，详见[用法示例](http://cn-static.udacity.com/mlnd/images/pprint.png)和[文档说明](https://docs.python.org/2/library/pprint.html#pprint.pprint)。

# In[70]:


from pprint import pprint
from decimal import Decimal, getcontext

getcontext().prec = 10

# 测试1.2 返回矩阵的行和列
print 'shape test1: should print "(3, 3)"'
print shape(A)
print 'shape test2: should print "(3, 4)"'
print shape(B)
print 'shape test3: should print "(0, 0)"'
print shape([])
        
# 测试1.3 每个元素四舍五入到特定小数数位
matxRoundTestData1 = [[Decimal('1.233'), Decimal('1.255'), Decimal('1.266'), Decimal('1.200'), Decimal('1.299')],
                      [Decimal('1.233'), Decimal('1.255'), Decimal('1.266'), Decimal('1.200'), Decimal('1.299')]]
matxRoundTestData2 = [[Decimal('1.233'), Decimal('1.255'), Decimal('1.266'), Decimal('1.200'), Decimal('1.299')],
                      [Decimal('1.233'), Decimal('1.255'), Decimal('1.266'), Decimal('1.200'), Decimal('1.299')]]
matxRound(matxRoundTestData1, 1)
matxRound(matxRoundTestData2, 2)
print 'matxRound test1: should print "[[1.2, 1.3, 1.3, 1.2, 1.3], [1.2, 1.3, 1.3, 1.2, 1.3]]"'
pprint(matxRoundTestData1)
print 'matxRound test2: should print "[[1.23, 1.25, 1.27, 1.2, 1.3], [1.23, 1.25, 1.27, 1.2, 1.3]]"'
pprint(matxRoundTestData2)

#TODO 测试1.4 计算矩阵的转置
print 'transpose test1: should print "[[1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12]]"'
pprint(transpose([[1,2,3],[4,5,6],[7,8,9],[10,11,12]]))
print 'transpose test2: should print "[[1, 5, 9], [2, 6, 10], [3, 7, 11], [4, 8, 12]]"'
pprint(transpose([[1,2,3,4],[5,6,7,8],[9,10,11,12]]))

#TODO 测试1.5 计算矩阵乘法AB，AB无法相乘
print 'matxMultiply test1: should print "None"'
pprint(matxMultiply([[1,2,3]], [[1,2,3]]))

#TODO 测试1.5 计算矩阵乘法AB，AB可以相乘
print 'matxMultiply test2: should print "[[22, 28], [49, 64]]"'
pprint(matxMultiply([[1,2,3], [4,5,6]],
                    [[1,2], [3,4], [5,6]]))


# # 2 Gaussign Jordan 消元法
# 
# ## 2.1 构造增广矩阵
# 
# $ A = \begin{bmatrix}
#     a_{11}    & a_{12} & ... & a_{1n}\\
#     a_{21}    & a_{22} & ... & a_{2n}\\
#     a_{31}    & a_{22} & ... & a_{3n}\\
#     ...    & ... & ... & ...\\
#     a_{n1}    & a_{n2} & ... & a_{nn}\\
# \end{bmatrix} , b = \begin{bmatrix}
#     b_{1}  \\
#     b_{2}  \\
#     b_{3}  \\
#     ...    \\
#     b_{n}  \\
# \end{bmatrix}$
# 
# 返回 $ Ab = \begin{bmatrix}
#     a_{11}    & a_{12} & ... & a_{1n} & b_{1}\\
#     a_{21}    & a_{22} & ... & a_{2n} & b_{2}\\
#     a_{31}    & a_{22} & ... & a_{3n} & b_{3}\\
#     ...    & ... & ... & ...& ...\\
#     a_{n1}    & a_{n2} & ... & a_{nn} & b_{n} \end{bmatrix}$

# In[13]:


# 构造增广矩阵，假设A，b行数相同
def augmentMatrix(A, b):
    resultMatix = [[] for i in range(len(A))]
    for index, row in enumerate(A):
        resultMatix[index].extend(row + b[index])
    return resultMatix


# ## 2.2 初等行变换
# - 交换两行
# - 把某行乘以一个非零常数
# - 把某行加上另一行的若干倍：

# In[14]:


# r1 <---> r2
# 直接修改参数矩阵，无返回值
def swapRows(M, r1, r2):
    M[r1], M[r2] = M[r2], M[r1]

# r1 <--- r1 * scale， scale!=0
# 直接修改参数矩阵，无返回值
def scaleRow(M, r, scale):
    if scale == 0:
        raise ValueError
    for index, value in enumerate(M[r]):
        M[r][index] = value * scale

# r1 <--- r1 + r2*scale
# 直接修改参数矩阵，无返回值
def addScaledRow(M, r1, r2, scale):
    cols = len(M[0])
    for col in range(cols):
        M[r1][col] = M[r1][col] + M[r2][col] * scale


# ## 2.3  Gaussian Jordan 消元法求解 Ax = b

# ### 提示：
# 
# 步骤1 检查A，b是否行数相同
# 
# 步骤2 构造增广矩阵Ab
# 
# 步骤3 逐列转换Ab为化简行阶梯形矩阵 [中文维基链接](https://zh.wikipedia.org/wiki/%E9%98%B6%E6%A2%AF%E5%BD%A2%E7%9F%A9%E9%98%B5#.E5.8C.96.E7.AE.80.E5.90.8E.E7.9A.84-.7Bzh-hans:.E8.A1.8C.3B_zh-hant:.E5.88.97.3B.7D-.E9.98.B6.E6.A2.AF.E5.BD.A2.E7.9F.A9.E9.98.B5)
#     
#     对于Ab的每一列（最后一列除外）
#         当前列为列c
#         寻找列c中 对角线以及对角线以下所有元素（行 c~N）的绝对值的最大值
#         如果绝对值最大值为0
#             那么A为奇异矩阵，返回None （请在问题2.4中证明该命题）
#         否则
#             使用第一个行变换，将绝对值最大值所在行交换到对角线元素所在行（行c） 
#             使用第二个行变换，将列c的对角线元素缩放为1
#             多次使用第三个行变换，将列c的其他元素消为0
#             
# 步骤4 返回Ab的最后一列
# 
# ### 注：
# 我们并没有按照常规方法先把矩阵转化为行阶梯形矩阵，再转换为化简行阶梯形矩阵，而是一步到位。如果你熟悉常规方法的话，可以思考一下两者的等价性。

# In[15]:


# 实现 Gaussain Jordan 方法求解 Ax = b

""" Gaussian Jordan 方法求解 Ax = b.
    参数
        A: 方阵 
        b: 列向量
        decPts: 四舍五入位数，默认为4
        epsilon: 判读是否为0的阈值，默认 1.0e-16
        
    返回列向量 x 使得 Ax = b 
    返回None，如果 A，b 高度不同
    返回None，如果 A 为奇异矩阵
"""

def gj_Solve(A, b, decPts=4, epsilon = 1.0e-16):
    if (len(A) != len(b)):
        return None
    Ab = augmentMatrix(A, b)

    for c in range(len(Ab[0]) - 1):
        maxValue = 0
        largestRow = -1
        for i in range(c, len(Ab)):
            if abs(Ab[i][c]) > abs(maxValue):
                maxValue = Ab[i][c]
                largestRow = i
        if abs(maxValue) < epsilon:
            return None
        else:
            swapRows(Ab, largestRow, c)
            
            scaleRow(Ab, c, 1.0 / maxValue)
            
            for i in range(len(Ab)):
                if i != c:
                    addScaledRow(Ab, i, c, -Ab[i][c])
        
    result = [[row[-1]] for row in Ab]
    return result


# ## 2.4 证明下面的命题：
# 
# **如果方阵 A 可以被分为4个部分: ** 
# 
# $ A = \begin{bmatrix}
#     I    & X \\
#     Z    & Y \\
# \end{bmatrix} , \text{其中 I 为单位矩阵，Z 为全0矩阵，Y 的第一列全0}$，
# 
# **那么A为奇异矩阵。**
# 
# 提示：从多种角度都可以完成证明
# - 考虑矩阵 Y 和 矩阵 A 的秩
# - 考虑矩阵 Y 和 矩阵 A 的行列式
# - 考虑矩阵 A 的某一列是其他列的线性组合

# TODO 请使用 latex （请参照题目的 latex 写法学习）
# 
# TODO 证明：
# 
# $ \text{使用单位矩阵 I 可以将 X 的第一列变换为全零，由于 Z 为零矩阵，做列变换时 Y 的第一列不受影响，仍为全零。}\\
# \text{由于变换后这一列所有元素为全零，是其他列的线性组合，所以矩阵 A 为奇异矩阵。}
# $

# ## 2.5 测试 gj_Solve() 实现是否正确

# In[76]:


# 构造 矩阵A，列向量b，其中 A 为奇异矩阵
A1 = [[3,2,2],
     [2,3,-2]]
b1 = [[1],
     [2]]

# 构造 矩阵A，列向量b，其中 A 为非奇异矩阵
A2 = [[1,0,0],
     [0,1,0],
     [0,0,1]]

b2 = [[3.0],
      [3.0],
      [3.0]]

# 求解 x 使得 Ax = b
x1 = gj_Solve(A1, b1)
x2 = gj_Solve(A2, b2)

print 'x1 should be None, actual value is: ', x1

# 计算 Ax

Ax2 = matxMultiply(A2, x2)

# 比较 Ax2 与 b

print 'Ax2 should equal to b2:'

print 'Ax2:', Ax2
print 'b2:', b2


# # 3 线性回归: 
# 
# ## 3.1 计算损失函数相对于参数的导数 (两个3.1 选做其一)
# 
# 我们定义损失函数 E ：
# $$
# E = \sum_{i=1}^{n}{(y_i - mx_i - b)^2}
# $$
# 
# 证明：
# $$
# \frac{\partial E}{\partial m} = \sum_{i=1}^{n}{-2x_i(y_i - mx_i - b)}
# $$
# 
# $$
# \frac{\partial E}{\partial b} = \sum_{i=1}^{n}{-2(y_i - mx_i - b)}
# $$
# 
# $$
# \begin{bmatrix}
#     \frac{\partial E}{\partial m} \\
#     \frac{\partial E}{\partial b} 
# \end{bmatrix} = 2X^TXh - 2X^TY
# $$
# 
# $$ 
# \text{其中 }
# Y =  \begin{bmatrix}
#     y_1 \\
#     y_2 \\
#     ... \\
#     y_n
# \end{bmatrix}
# ,
# X =  \begin{bmatrix}
#     x_1 & 1 \\
#     x_2 & 1\\
#     ... & ...\\
#     x_n & 1 \\
# \end{bmatrix},
# h =  \begin{bmatrix}
#     m \\
#     b \\
# \end{bmatrix}
# $$

# TODO 请使用 latex （参照题目的 latex写法学习）
# 
# TODO 证明：
# 
# $$
# \frac{\partial E}{\partial m} = \sum_{i=1}^{n}{2(y_i - mx_i - b)\frac{\partial {(y_i - mx_i - b)}}{\partial m}} = \sum_{i=1}^{n}{2(y_i - mx_i - b)(-x_i)} = \sum_{i=1}^{n}{-2x_i(y_i - mx_i - b)}
# $$
# 
# $$
# \frac{\partial E}{\partial b} = \sum_{i=1}^{n}{2(y_i - mx_i - b) \frac{\partial {(y_i - mx_i - b)}}{\partial b}} = \sum_{i=1}^{n}{2(y_i - mx_i - b)(-1)} = \sum_{i=1}^{n}{-2(y_i - mx_i - b)}
# $$
# 
# $$
# 2X^TXh - 2X^TY = -2X^T(Y-Xh) = -2X^T\begin{bmatrix}
#     y_1 - mx_1 - b \\
#     y_2 - mx_2 - b \\
#     ... \\
#     y_n - mx_n - b \\
# \end{bmatrix} = -2\begin{bmatrix}
#     \sum_{i=1}^{n}{x_i(y_i - mx_i - b)} \\
#     \sum_{i=1}^{n}{y_i - mx_i - b}
# \end{bmatrix} = \begin{bmatrix}
#     \frac{\partial E}{\partial m} \\
#     \frac{\partial E}{\partial b} 
# \end{bmatrix}
# $$

# ## 3.1 计算损失函数相对于参数的导数（两个3.1 选做其一）
# 
# 证明：
# 
# $$
# E = Y^TY -2(Xh)^TY + (Xh)^TXh
# $$ 
# 
# $$
# \begin{bmatrix}
#     \frac{\partial E}{\partial m} \\
#     \frac{\partial E}{\partial b} 
# \end{bmatrix}  = \frac{\partial E}{\partial h} = 2X^TXh - 2X^TY
# $$
# 
# $$ 
# \text{其中 }
# Y =  \begin{bmatrix}
#     y_1 \\
#     y_2 \\
#     ... \\
#     y_n
# \end{bmatrix}
# ,
# X =  \begin{bmatrix}
#     x_1 & 1 \\
#     x_2 & 1\\
#     ... & ...\\
#     x_n & 1 \\
# \end{bmatrix},
# h =  \begin{bmatrix}
#     m \\
#     b \\
# \end{bmatrix}
# $$

# TODO 请使用 latex （请参照题目的 latex 写法学习）
# 
# TODO 证明：

# ## 3.2  线性回归
# 
# ### 求解方程 $X^TXh = X^TY $, 计算线性回归的最佳参数 h

# In[65]:


# 实现线性回归
'''
参数：(x,y) 二元组列表
返回：m，b
'''
def linearRegression(points):
    xlist = []
    ylist = []
    
    for index, point in enumerate(points):
        xlist.append([point[0], 1])
        ylist.append([point[1]])
                
    xT = transpose(xlist)
    
    # print 'x', xlist
    
    xTx = matxMultiply(xT, xlist)
    
    # print 'xT', xT
    
    # print 'xTx: ', xTx
    
    # Inverse
    rows = len(xTx)
    inv_xTx = [[] for i in range(rows)]
    for curCol in range(rows):
        col = []
        for curRow in range(rows):
            if curCol == curRow:
                col.append([1])
            else:
                col.append([0])
        invCol = gj_Solve(xTx, col)
        for row in range(rows):
            inv_xTx[row].append(invCol[row][0])
    
    # print 'inv', inv_xTx
    
    inv_xTx_xT = matxMultiply(inv_xTx, xT)
    
    # print 'inv_xTx_xT', inv_xTx_xT
    
    # print 'ylist', ylist
    
    inv_xTx_xT_y = matxMultiply(inv_xTx_xT, ylist)
    
    # print 'inv_y', inv_xTx_xT_y
    
    return inv_xTx_xT_y[0][0], inv_xTx_xT_y[1][0]


# ## 3.3 测试你的线性回归实现

# In[80]:


# 构造线性函数
def f(x, m = 1, b = 0):
    return m * x + b

# 构造 100 个线性函数上的点，加上适当的高斯噪音
import random

m = 3.0
b = 2.0

points = []

for x in range(3):
    y = f(x, m, b) + random.gauss(0, 0.5)
    points.append((x, y))

#TODO 对这100个点进行线性回归，将线性回归得到的函数和原线性函数比较

rm, rb = linearRegression(points)

print 'origin m, b is:', m, b
print 'regression m, b is:', rm, rb


# ## 4.1 单元测试
# 
# 请确保你的实现通过了以下所有单元测试。

# In[81]:


import unittest
import numpy as np

from decimal import *

class LinearRegressionTestCase(unittest.TestCase):
    """Test for linear regression project"""

    def test_shape(self):

        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.randint(low=-10,high=10,size=(r,c))
            self.assertEqual(shape(matrix.tolist()),(r,c))


    def test_matxRound(self):

        for decpts in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()
            dec_true = [[Decimal(str(round(num,decpts))) for num in row] for row in mat]

            matxRound(mat,decpts)
            dec_test = [[Decimal(str(num)) for num in row] for row in mat]

            res = Decimal('0')
            for i in range(len(mat)):
                for j in range(len(mat[0])):
                    res += dec_test[i][j].compare_total(dec_true[i][j])

            self.assertEqual(res,Decimal('0'))


    def test_transpose(self):
        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()
            t = np.array(transpose(mat))

            self.assertEqual(t.shape,(c,r))
            self.assertTrue((matrix.T == t).all())


    def test_matxMultiply(self):

        for _ in range(10):
            r,d,c = np.random.randint(low=1,high=25,size=3)
            mat1 = np.random.randint(low=-10,high=10,size=(r,d)) 
            mat2 = np.random.randint(low=-5,high=5,size=(d,c)) 
            dotProduct = np.dot(mat1,mat2)

            dp = np.array(matxMultiply(mat1,mat2))

            self.assertTrue((dotProduct == dp).all())


    def test_augmentMatrix(self):

        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            A = np.random.randint(low=-10,high=10,size=(r,c))
            b = np.random.randint(low=-10,high=10,size=(r,1))

            Ab = np.array(augmentMatrix(A.tolist(),b.tolist()))
            ab = np.hstack((A,b))

            self.assertTrue((Ab == ab).all())

    def test_swapRows(self):
        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()

            r1, r2 = np.random.randint(0,r, size = 2)
            swapRows(mat,r1,r2)

            matrix[[r1,r2]] = matrix[[r2,r1]]

            self.assertTrue((matrix == np.array(mat)).all())

    def test_scaleRow(self):

        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()

            rr = np.random.randint(0,r)
            with self.assertRaises(ValueError):
                scaleRow(mat,rr,0)

            scale = np.random.randint(low=1,high=10)
            scaleRow(mat,rr,scale)
            matrix[rr] *= scale

            self.assertTrue((matrix == np.array(mat)).all())
    
    def test_addScaleRow(self):

        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()

            r1,r2 = np.random.randint(0,r,size=2)

            scale = np.random.randint(low=1,high=10)
            addScaledRow(mat,r1,r2,scale)
            matrix[r1] += scale * matrix[r2]

            self.assertTrue((matrix == np.array(mat)).all())


    def test_gj_Solve(self):

        for _ in range(10):
            r = np.random.randint(low=3,high=10)
            A = np.random.randint(low=-10,high=10,size=(r,r))
            b = np.arange(r).reshape((r,1))

            x = gj_Solve(A.tolist(),b.tolist())
            if np.linalg.matrix_rank(A) < r:
                self.assertEqual(x,None)
            else:
                # Ax = matxMultiply(A.tolist(),x)
                # print A
                # print b
                Ax = np.dot(A,np.array(x))
                loss = np.mean((Ax - b)**2)
                # print Ax
                # print loss
                self.assertTrue(loss<0.1)


suite = unittest.TestLoader().loadTestsFromTestCase(LinearRegressionTestCase)
unittest.TextTestRunner(verbosity=3).run(suite)


# In[ ]:





# In[ ]:




