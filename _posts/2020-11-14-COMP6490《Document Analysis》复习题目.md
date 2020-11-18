## 计算

1. TF-IDF、cos
2. recall、precision、F1-Score
3. HITS and PageRank 
4. Matrix
5. derivative
6. PMI
7. Backprop
8. attention
9. transformer
10. K-means
11. derivation for the string
12. language model probality
13. Nivre’s arc-eager parser 

## 3. IR

#### Q1  Pre-processing 

**(a)** **List all of the main steps used in IR system pre-processing in the order they are performed. Briefly describe each step.** 

Answer: 

Tokenization, stop word removal, normalisation 

**(b) Describe the difference between lemmatization and stemming.** 

Answer: 

Stemming always removes some portion from the end of words, lemmatization can replace words entirely e.g. good -> better. Usually, lemmatization takes longer to compute. 

**(c) Apart from whitespace tokenization, come up with 2 other methods of tokenization. Discuss why each of your proposed tokenization methods might be useful for the task of IR.**

Answer: 

Tokenize based on grammar (e.g. treat all of ‘,.!?’ As whitespace). 

Use regexp to parse numbers (e.g. 1,000) into a single token. 



#### Q2: Querying 

Consider the following term-document matrix for 3 terms "quick","brown","fox" in a collection of 3 documents:

![image-20201112103524204](/Users/lizhuoran/Library/Application Support/typora-user-images/image-20201112103524204.png)

Tf = 该词在文章出现的次数

IDF = log（文档总数/包含该词的文档数）

TF-IDF = TF * IDF

**(a) calculate the tf-idf score**

![image-20201112103631997](/Users/lizhuoran/Library/Application Support/typora-user-images/image-20201112103631997.png)

**(b)Now suppose that a user runs the query “quick fox”. Calculate the cosine similarity between this query and each of the 3 documents, where the document and query vectors are given by the tf-idf score of each term. Which document is retrieved first?** 

$tf-idf_q$ = [log3, 0 ,0]

cos = AB/(|A||B|)

Answer: 

Sim(query, Doc1) = log3 * 3log3 / (log3 * 3log3) = 1 

Sim(query, Doc2) = 0 

Sim(query, Doc3) = 0 

1 is retrieved first. 

**(c)** **Write down the inverted index that would be created from this term-document matrix as a Python dictionary.** 

Answer: 

{“quick”: [(Doc1, 3)], 

“brown”: [(Doc2, 1), (Doc3, 3)], 

“fox”: [(Doc1, 2), (Doc2, 1), (Doc3, 6)]} 

**(d)** **Explain the importance of the idf component of the tf.idf score. How does the idf change the weights of rare terms and why is this useful in information retrieval?** 

Answer: 

The IDF term weights rare words higher, whereas words that appear in lots of documents are given small weights. This is useful for IR because a word that appears in only a few documents is likely related to the topic of those documents, which means they will inform the query results more. 



#### Q3: Evaluation 

Suppose that we are evaluating our IR system. For a given query, our system retrieves 10 documents, which are marked as being relevant (R) or irrelevant (I) in the following order: 

R, I, R, R, I, R, R, R, R, R 

The list is ordered left to right, so the leftmost R is the relevance of the first retrieved document. There are 12 relevant documents in the entire collection. 

tp = 8, fp = 2, fn = 4, tn = 

**(a)** **Calculate the recall at 5 documents retrieved.** 

Tp = 3,fp=2,fn = 9

recall = tp/(tp+fn)

Precision = tp/(tp+fp)

Accuracy = (tp+tn)/(tp+tn+fp+fn)

Recall = 3/12

P = 3/5

**(b) Calculate the interpolated precision at 20% recall.** 



**(c)** **Calculate the F1-score at 5 documents retrieved** 

F1 = 2PR/(P+R)

F1 = 0.35294118

**(d)** **Consider the task of building an IR system for a collection of legal documents (patents, court transcripts, etc) to be used by legal firms. How would you evaluate this IR system? Compare the differences in user needs between this system and a typical web search application. How would these differences influence what metrics you use to measure your system performance?** 

Answer: 

For web search, users just want to find some relevant results as quickly as possible and are unlikely to look through more than 5 retrieved documents. A lawyer preparing for a court case might need to find some contract or other important document relevant to the case. Because of this they would be willing to spend more time looking through results as it is very important that they find the specific document they are looking for. Therefore, recall@100 would be a good measure for the system, as it captures how well the system can eventually find relevant documents. 

## 4. Web Search & Linear Regression

####  Q1: HITS and PageRank 

**(a)** **Give 2 examples of differences between bibliographic references and webpage links and explain how these differences affect IR algorithms for each.** 

Answer: 

1. Many webpage links are purely navigational. 

2. Many webpages with high in-degree are portals not content providers. 



This means that web search algorithms should try and separate content providers from hubs and ignore links within the same website. 

**(b)** **Given the following graph which describes hyperlinks between 5 web-pages:** 

 **Run the HITS algorithm for two iterations. What are the authority and hub scores for each website?**  

![image-20201110152132067](/Users/lizhuoran/Library/Application Support/typora-user-images/image-20201110152132067.png)

Answer: 

a0 = [1, 1, 1, 1, 1], h0 = [1, 1, 1, 1, 1] 

a1 = [1, 1, 1, 2, 1] / sqrt(8), h1 = [3, 1, 1, 1, 0] / sqrt(12) 

a2 = [1, 3, 3, 2, 3] / (sqrt(12) * sqrt(32/12)), h2 = [3, 2, 2, 1, 0] / (sqrt(8) * sqrt(18/8)) 

**(c)** **Using the same graph from question (b), run the Page Rank algorithm for two iterations. What are the authority scores for each website?** 

Answer: 

r0 = [1, 1, 1, 1, 1] / 5 

r1 = [1/5, 1/15, 1/15, 2/5, 1/15] / (12/15) 

r2 = [1/2, 1/12, 1/12, 1/6, 1/12] / (11/12) 

#### Q2: One-hot Vectors 

**Suppose we have three words in our language: “quick”, “brown”, “fox”. Compute one-hot vector representations for each of these 3 words.** 

Answer: 

“quick” = [1 0 0] 

“brown” = [0 1 0] 

“fox” = [0 0 1] 

#### Q3: Matrix Algebra 

**(a)** **Compute AB, where 𝐴=[2 −1 0 1] and 𝐵=[−1 3 1 −2]** 

Answer: [−3 8 1 −2] 

**(b)** **Compute ‖𝑥−𝑦‖2, where 𝑥=[2−1], 𝑦=[−13]** 

Answer: 5 

**(c)** **Compute 𝑑𝐿𝑑𝑊 where 𝑊=[𝑤1𝑤2], 𝑥=[2−1], 𝐿=12(𝑊𝑥−1)2** 

Answer: 𝑑𝐿𝑑𝑤1=(𝑊𝑥−1)2,𝑑𝐿𝑑𝑤2=(𝑊𝑥−1)(−1) 

#### Q4: Multiple Linear Regression 

**(a)** **Suppose we are trying to build a linear model which predicts 3 output variables for each 1-dimensional input data point. One approach would be to train one [3,1] weight matrix which maps inputs to a vector output. Another approach could be to train 3 separate [1,1] weight matrixes, one for each of the target variables. Which one of these methods would you expect to perform better and why?** 

Answer: They are exactly the same. 

#### Q5: Adding more features 

**When using linear regression, we must represent our input objects as a vector. Each component of the vector is a feature which describes something about the input (e.g. for TF vectors, each component tells the model how many times a particular word occurs in the document). Is it possible that adding more features to a dataset, that is giving more information about each object to the model, causes a model to make worse predictions? Justify why you believe this is possible or not possible.** 

Answer: It is possible, consider the following 2 datasets, where the true relationship is 𝑦=𝑥1+0𝑥2 (with some random noise). 

D1 = {([0,0],-1), ([0,0],1), ([1,0],0), ([1,0],2)} 

D2 = {([0,0],-1), ([0,1],1), ([1,0],0), ([1,1],2)} 

Solving for the line with the least squared error on D1 gives 𝑦=𝑥1+0𝑥2, but on D2 gives 𝑦=𝑥1−2𝑥2. Even though there is no relationship between 𝑥2 and 𝑦, if we introduce 𝑥2 into the dataset, the model may learn a relationship that does not exist due to noise in the dataset. This is known as overfitting. 

#### Q6: Cross-Entropy Loss (Challenge question) 

**Suppose we are training a linear classifier. For some training datapoint we have 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(𝑊𝑥+𝑏)=[0.10.9] 𝑦=[10]** 

**(a) Compute the derivative of the loss of this datapoint with respect to the first bias 𝑑𝐿𝑑𝑏1, where the loss is mean squared error 𝐿=Σ12(𝑠𝑜𝑓𝑡𝑚𝑎𝑥(𝑊𝑥+𝑏)𝑖−𝑦𝑖)22𝑖=1.** 

Answer: 

Let 𝑧=𝑠𝑜𝑓𝑡𝑚𝑎𝑥(𝑊𝑥+𝑏), then 𝑑𝐿𝑑𝑏1=𝑑𝐿𝑑𝑧1𝑑𝑧1𝑑𝑏1+𝑑𝐿𝑑𝑧2𝑑𝑧2𝑑𝑏1 𝑑𝐿𝑑𝑧1=(z1−y1) 

𝑑𝐿𝑑𝑧2=(z2−y2) 𝑑𝑧1𝑑𝑏1=𝑑exp((𝑊𝑥+𝑏)1)exp((𝑊𝑥+𝑏)1)+exp((𝑊𝑥+𝑏)2)𝑑𝑏1 𝑑𝑧1𝑑𝑏1=exp((𝑊𝑥+𝑏)1)exp((𝑊𝑥+𝑏)1)+exp((𝑊𝑥+𝑏)2)−exp((𝑊𝑥+𝑏)1)2(exp((𝑊𝑥+𝑏)1)+exp((𝑊𝑥+𝑏)2))2 =𝑠𝑜𝑓𝑡𝑚𝑎𝑥(𝑊𝑥+𝑏)1(1−𝑠𝑜𝑓𝑡𝑚𝑎𝑥(𝑊𝑥+𝑏)1) 𝑑𝑧2𝑑𝑏1=𝑠𝑜𝑓𝑡𝑚𝑎𝑥(𝑊𝑥+𝑏)2(1−𝑠𝑜𝑓𝑡𝑚𝑎𝑥(𝑊𝑥+𝑏)2) 

**(b) Recompute (a) but this time use cross-entropy loss instead of mean squared error, 𝐿=Σlog (𝑠𝑜𝑓𝑡𝑚𝑎𝑥(𝑊𝑥+𝑏)𝑖)𝑦𝑖2𝑖=1=log (𝑠𝑜𝑓𝑡𝑚𝑎𝑥(𝑊𝑥+𝑏)1)** 

Answer: 𝑑𝐿𝑑𝑏1=𝑑log(𝑧1)𝑑𝑧1𝑑𝑧1𝑑𝑏1 𝑑log(𝑧1)𝑑𝑧1=1𝑧1 𝑑𝑧1𝑑𝑏1=𝑠𝑜𝑓𝑡𝑚𝑎𝑥(𝑊𝑥+𝑏)1(1−𝑠𝑜𝑓𝑡𝑚𝑎𝑥(𝑊𝑥+𝑏)1) 

**(c) If we were to train our model using one of these losses, which of these two losses do you think would result in a better model?** 

Answer: 

The cross-entropy derivative can be arbitrarily large, while the MSE gradient is the sum of two values bounded between 0 and 1, so the cross-entropy derivative should allow for much faster training in general. 

```
for the cross-entropy, the highest error between predict and label, the faster the decline
```

## 5. Embedding_DNN

#### Q1: Pairwise Mutual Information Vectors 

**Given the following word cooccurrence matrix** 

 **(a)** **Compute the pairwise mutual information representation vector of the words “quick”, “brown”, “fox”.** 

![image-20201109164150737](/Users/lizhuoran/Library/Application Support/typora-user-images/image-20201109164150737.png)

Answer: 

Quick = [log((4/30)/((1/3)*(1/3))), log((3/30)/((1/3)*(12/30))), log((3/30)/((1/3)*(8/30)))] 

Brown = [log((4/30)/((11/30)*(1/3))), log((6/30)/((11/30)*(12/30))), log((1/30)/((11/30)*(8/30)))] 

Fox = [log((2/30)/((9/30)*(1/3))), log((1/10)/((9/30)*(12/30))), log((4/30)/((9/30)*(8/30)))] 

 **(b)** **When calculating pairwise mutual information, we take the logarithm of the ratio of probabilities. Describe what effect applying the logarithm has on the vector representations.** 

Answer: It reduces the size of the components 

```
It can reduce the value of ratio largely.
```

#### Q2: Word2vec 

**(a)** **Given the following sentence: “The quick brown fox jumps”, we wish to perform word2vec embedding with a window size of 3 to create a representation vector of each word. Transform this sentence into a training dataset that we can train our embedding model on. Your dataset should have an input vector and a target vector for each context in the sentence.** 

 Answer: 

D = {([the brown], quick), 

([quick fox], brown), 

([brown jumps], fox)} 

 **(b)** **In your own words explain why word2vec models tend to learn vector representations which obey simple arithmetic operations, e.g. king – man + woman = queen.** 

Answer: The word2vec model tries to predict a missing word based on its context using a *linear* function. In order for it to make accurate predictions, then it must be able to encode semantic meaning of words into *linear* operations on vectors. 

```
Kings and queens often appear together, and men and women also appear together, which belong to association words. Therefore, the regular and stable position of a word is usually determined by its combination words, which are words that often appear together. So these most frequently occurring combination words determine the update rule of the target word.
```

**(c)** **If you replaced the linear predictor in word2vec with a 1 hidden layer neural network, would the model still learn vector representations which obey simple arithmetic operations? Would the learned vector representations be more useful for other models?** 

Answer: No, with a NN predictor the meaning of words could, and probably would, be encoded in complicated non-linear operations. 

#### Q3: Backprop 

![image-20201112211559253](/Users/lizhuoran/Library/Application Support/typora-user-images/image-20201112211559253.png)

#### Q4: Multiple Outputs 

**(a)** **Suppose we are trying to build a neural network model which predicts 3 output variables for each 5-dimensional input data point. One approach would be to train one neural network which outputs a 3-dimensional vector. Another approach could be to train 3 separate neural networks, one for each of the target variables. Which one of these methods would you expect to perform better and why?** 

Answer: For the single neural network, its hidden layers will learn to create a representation of the input from which all 3 values can be predicted. With 3 different neural networks, each one can learn different representations. This will usually result in a regularizing effect on the single network, since it uses fewer parameters. 

单个网络具有相同的隐藏层表示，所以结果更正规。

#### Q5: Generalization 

**(a)** **Consider a 1 hidden layer neural network. Describe what effect the number of hidden neurons has on the model’s predictive ability**. 

Answer: For a neural network to perform well, it must have enough hidden neurons to be able to approximate the true distribution of the data. However, if a neural network has many hidden neurons then it can end up learning noise (small random variations) from the training dataset and hence make worse predictions than one with less neurons. 

模型需要大量神经元，但是过多会引起过拟合。

**(b)** **Describe what effect the activation function has on the model’s predictive ability.** 

Answer: The output of a neural network is made by scaling, shifting, and flipping the activation function. For a given dataset, some activation functions may be better suited than others, but it is usually difficult to know which, hence simple activation functions (such as relu) are preferred.

不同的训练集使用不同的激活函数可能效果会很好，但是很难去选择，所以一般使用简单有效的。 

**(c)** **Consider a one hidden layer neural network with 10 hidden neurons and a two hidden layer network with 5 and 2 hidden neurons. Both networks have 1 dimensional input. Which of these networks has a greater representation capacity (that is, will be able to represent the most complicated functions)? In general, how does the capacity of a network scale with the number of layers?** 

Answer: Consider the 2 layer network. Each neuron in the second layer can be though of as a 1 layer network with 5 hidden neurons. Then the output of the network is a linear combination of 2 5 hidden neuron networks, which is the same as one layer of 10 neurons. This ignores the fact that for the 2 layer network, the second layer neurons have to use the *same* hidden neurons, so actually it will have slightly less capacity. In general, assuming that each layer has more neurons than the input dimension, capacity grows exponentially in depth. 

2*5微弱。如果神经元个数大于输入维度，则层数增加能力增强很多。

#### Q6: Early Stopping 

**Neural networks are almost always trained with “early stopping”, this means that during training the network is regularly evaluated on a held-out *validation* dataset, and if the performance on the *validation* set does not increase then the training process is stopped early.** 

**(a)** **Do you think that early stopping would improve or decrease a neural network’s predictive performance?** 

Answer: Prevents overfitting, improving performance. 

提升，防止过拟合

**(b)** **If you split your dataset into 2 parts, train and test, and used the test set to decide when to stop training, do you think that the test accuracy would still be a good estimation of how the network will perform on new unseen data?** 

Answer: No, since we have specifically optimized the network to perform well on this test set. 

否，因为我们已经对这个测试集进行了优化。

**(c)** **If we are using early stopping, how do you think the test accuracy will change as we increase the number of hidden neurons?** 

Answer: With early stopping, performance to increase as the number of hidden neurons increases, seemingly without limit. 

性能随着神经元数目增加，似乎没有限制。

## 6. RNN

####  Q1: Recurrent models 

**In your own words explain why recurrent models are needed to process sequential data.** 

Answer: Sequences can have arbitrary length. 

由于序列长度可以任意。

#### Q2: Parameter counts 

**a) For a simple RNN cell with input dimension 𝑖 and hidden dimension ℎ, how many parameters (weights and biases) does this cell have?** 

Answer: (i+h)*h + h 

**b) How many parameters does a GRU cell with input dimension 𝑖 and hidden dimension ℎ have?** 

Answer: 3 * ((i+h)*h + h) 

**c) How many parameters does a LSTM cell with input dimension 𝑖 and hidden dimension ℎ have?** 

Answer: 4 * ((i+h)*h + h) 

#### Q3: Vanishing and exploding gradients 

**For this question we will attempt to understand why gradients vanish and explode in RNNs. To make the calculations simpler we will ignore the input vector at each timestep as well as the bias, so the update equation is given by** 

![image-20201112214722690](/Users/lizhuoran/Library/Application Support/typora-user-images/image-20201112214722690.png)

![image-20201112214753112](/Users/lizhuoran/Library/Application Support/typora-user-images/image-20201112214753112.png)

**b) Explain why the value of this derivative will become very small (vanish) or very large (explode) as the number of time-steps 𝑡 increases.** 

Answer: Ignoring the activation function, this derivative is proportional to 𝑊𝑡. If the weights in W are less than 1 this quantity shrinks exponentially, if they are greater than 1 it grows exponentially. 

梯度与权重W成比例，大于1指数增长，小于1指数收缩

**c) Explain why vanishing and exploding gradients mean that the model will “forget”.** 

Answer: During training, gradients of one time-step w.r.t to previous time-steps will be either small or large, meaning weights will either be not changed or not converge. This un-stable training means that previous hidden states will not learn to be useful for the current step. 

因为如果梯度很大或很小，则权重可能不会改变，所以隐藏层无法学习。

**d) In our calculations we ignored the inputs at each step and the bias. If we use the full update equation ℎ𝑡+1 = 𝜙(𝑊1ℎ𝑡 + 𝑊2𝑥𝑡 + 𝑏), do our conclusions about vanishing and exploding gradients still hold?** 

Answer: Yes, because the gradient is still proportional to 𝑊𝑡. 

**e) We have seen that the LSTM and GRU cells can help to avoid exploding and vanishing gradients. Can you think of any other ways to change a simple RNN in order to increase its memory?** 

(1) Previous output of  t-1 timestep as  memory content to the input of  t titimestep

(2) The bidirectional RNN is composed of a forward RNN and a reverse RNN. The forward RNN reads the input sequence from front to back, and the reverse RNN reads the input sequence from back to forward. The network output at each moment is composed of the output of the forward RNN and The output of the reverse RNN is jointly determined

## 7. attention

#### Q1: Encoder-Decoder models 

**We’ve seen how encoder-decoder architectures can be used for natural language translation. Give another example of a task that encoder-decoder models would be useful for.** 

Answer: Any task where the output is some type of data structure, e.g. translating code from one programming language to another, semantic parsing, predicting how a network (graph) will change over time. 

输出为某种数据结构类型的任何任务

#### Q2: Attention practice 

**Given a set of key vectors {𝑘1=[2−13],𝑘2=[124],𝑘3=[−2−30]} and query vector 𝑞=[1−12], compute the attention weighted sum of keys using dot-product similarity 𝑠𝑖𝑚(𝑥,𝑦)=𝑥𝑇𝑦** 

𝑎̃ = [k*q]

$a = e^𝑎̃_i/sum(e^𝑎̃)$

Answer: 𝑎̃=[9,7,1], 𝒂=[0.88,0.12,0],𝑦=0.88[2−13]+0.12[124]+0[−2−30]=[1.88−0.643.12] 

####  Q3: Similarity scores 

**A scoring function is used to measure similarity between two vectors, which then determines the attention weight between query and key. Compare and contrast each of the different scoring functions given in the lecture slides. How do you think these differences would affect an attention model?** 

Answer: 

The scaled dot-product attention makes the expected size of the scores independent of the vector dimension, which leads to more stable gradients across layers. 𝑥𝑇𝑊𝑦 allows for different dimensions to contribute different amounts to the resulting score. 𝜈𝑇tanh⁡(𝑊[𝑥,𝑦]) is training a 1 hidden layer neural network to predict the similarity of the two vectors x and y, it can learn a non-linear similarity function which allows for more expressivity than the simple dot product similarities, at the cost of more parameters.

缩放点积使得分的期待大小独立于向量维度，梯度更加稳定。

 𝑥𝑇𝑊𝑦 允许不同维度对结果贡献不同。

𝜈𝑇tanh⁡(𝑊[𝑥,𝑦]) 使用1层隐藏层预测相似性，可以学习非线性的相似性函数，表达更强，要用更多的参数。

#### Q4: Distance scores 

**Usually attention weights are computed based on similarity between query and key. Would it be possible to instead compute attention weights based on the Euclidean distance between query and key, so that vectors which are further apart have smaller weight? If so, then explain how you would do it. If not, then explain what trouble you would run into.** 

Answer: Usually neural network hidden spaces tend to encode meaning as directions in the space, so if you have 2 vectors which both pointed in the same direction but one was much longer, they would have a large distance between them but both would tend to represent the kind of input. For this reason, distances tend to not work as well. 

不能，距离代替相似度，相似类型的表示会有很大距离。

#### Q5: SoftMax normalisation 

**a) Attention weights are run through a SoftMax function to ensure they sum up to 1. Explain why this step is important.** 

Answer: The SoftMax ensures that the weights add up to 1, which means the weights are split among the items. This forces the model to focus its attention, since having a large weight for one item means smaller weights for the others. 

强制让模型注意到某部分，大的权重意味着其他就是小权重。

**b) Would the attention mechanism still work if we did the normalisation as follows?** 

𝒂𝑖=𝒓𝒆𝒍𝒖(𝒂̃𝒊)Σ𝒓𝒆𝒍𝒖(𝒂̃𝒋)𝒏𝒋=𝟏 

Answer: in principle it should still work, although you might run into issues if the denominator is 0. 

原则可以，尽管分母可能为0.

#### Q6: Tree based Encoder-Decoder (Challenge Question) 

**Suppose that we have a dataset where both the inputs and outputs are binary trees. Each node in the tree contains a vector of features which represent that tree. The predicted trees should have the same structure as the target trees, and all of the predicted nodes should have the same vectors as the target nodes.** 

**a) Describe how you would design an encoder-decoder model for this dataset.** 

Answer: The encoder can be a RNN which maps a node and both of its children to a vector. This encoder can be applied to every node in the tree, beginning with the leaf nodes and working up to the root. The decoder would work in reverse, it maps a vector to 3 output vectors, the first is used to predict the item of the current node, and the other 2 are used for the children. 

```
该编码器可以应用于树中的每个节点，从叶节点开始一直到根。解码器将反向工作，它将一个向量映射到3个输出向量，第一个用于预测当前节点的项目，其他两个用于子节点
```

**b) How would your model need to change if the trees were not binary, so that each node could have any number of children?** 

Answer: In the encoder we would need to apply the RNN across all of the child nodes first, before combining child and parent nodes together. Similarly, the decoder would decode all of the children nodes as a list. 

```
在编码器中，在将子节点和父节点组合在一起之前，我们需要先在所有子节点上应用RNN。同样，解码器会将所有子节点解码为列表。
```

## 8. Transformer

####  Q1: Transformer datasets 

**Describe a type of task where a transformer model is likely to perform much better than a recurrent model.** 

Answer: Any task where long term dependency is needed, e.g. generating news articles, question answering, language translation. 

```需要长期依赖的任何任务
需要长期依赖的任何任务
```

#### Q2: Transformer complexity 

**a) What is the computational complexity (how many computation steps do you need to perform) for a self-attention block with vector dimension 𝑑 and sequence length 𝑙** 

![image-20201112222330624](/Users/lizhuoran/Library/Application Support/typora-user-images/image-20201112222330624.png)

**b) Is this complexity smaller or larger compared to a GRU with dimension 𝑑 and sequence length 𝑙** 

Answer: For a GRU, each step only requires computing 3 gates, which is O(d\*d*l). The GRU has a lower complexity. 

**c) Hence the transformer model can be trained in much shorter time, True or False?** 

Answer: While the GRU needs to perform fewer operations, a transformer can perform all of its computation in parallel. So provided you have many processors to run on, a transformer can actually be computed faster (in less wall time). 

GRU操作虽然少，但是transformer可以并行计算编码器。

#### Q3: Self-Attention practice 

![image-20201112222602053](/Users/lizhuoran/Library/Application Support/typora-user-images/image-20201112222602053.png)

#### Q4: Feed-forward layers 

**The transformer architecture repeats self-attention layers and then feed-forward layers. Explain the importance of each of these layers.** 

Answer: The self-attention layers mix up the representations of each input word, each input word can attend to every other word in the input, regardless of distance, to determine what to include in its own representation. The feed-forward layers introduce non-linear processing of each word’s representation. 

```
每个输入单词都可以与输入中的每个其他单词相关联，而不论距离如何，FF曾引入了非线性处理。
```

#### Q5: Pre-training 

**Explain why neural network architectures are especially well suited for pre-training on un-labelled data.** 

Answer: Neurons in a neural network tend to learn useful representations of their input. The neurons in lower layers will often learn general features that are useful for a wide range of tasks, not just the one they are trained on. Additionally, neural networks can be trained iteratively, unlike some other machine learning models which need to be fit on the entire dataset. 

```
神经网络中的神经元倾向于学习其输入的有用表示。较低层的神经元通常会学习对许多任务有用的一般特征，而不仅仅是对它们进行训练的特征。此外，神经网络可以迭代地训练，这与其他一些需要适合整个数据集的机器学习模型不同
```

#### Q6: Self-supervised objectives 

**a) In BERT’s masked language modelling training objective, masked tokens are sometimes kept as the same word or replaced with a random word, instead of using the [MASK] token. Explain why this is done.** 

Answer: Always training with the [MASK] token means that the model will learn to expect [MASK] tokens in its input, however at test time there will not be any [MASK] tokens in the input. By sometimes training with regular words, the model learns to represent all of the words in the input. 

```
始终使用[MASK]令牌进行训练意味着该模型将学会在其输入中期望使用[MASK]令牌，但是在测试时，输入中将没有任何[MASK]令牌。通过有时使用规则的单词进行训练，该模型学会表示输入中的所有单词
```

**b) Explain why for BERT’s next sentence prediction task, inputs are encoded as [CLS] sentence1 [SEP] sentence2 [SEP].** 

Answer: The [CLS] token lets the model know that it is performing classification (and not trying to fill in [MASK] missing words), and the [SEP] token lets the model know where one sentence ends and the next begins. 

```
[CLS]令牌使模型知道它正在执行分类（而不是尝试填充[MASK]丢失的单词），而[SEP]令牌使模型知道一个句子在哪里结束而下一句话在哪里开始。
```

## 9. clustering and semantics

#### Q1: Unsupervised Learning 

**a) Discuss the differences between supervised learning, self-supervised learning and unsupervised learning.** 

Answer: Supervised learning has human provided labels, self-supervised has labels derived from the raw data itself, unsupervised learning uses no labels what-so-ever. 

```
监督学习具有人工提供的标签，自我监督具有源自原始数据本身的标签，无监督学习则根本不使用标签。
```

**b) Does unsupervised learning require a human to provide any additional information, or does it only require a set of data points?** 

Answer: Unsupervised learning typically requires some additional information about the structure of the data space, for example clustering algorithms require the user to specify a distance metric. 

```
无监督学习通常需要一些有关数据空间结构的附加信息，例如，聚类算法要求用户指定距离度量。
```

#### Q2: Types of Clustering 

**a) Discuss the differences between flat and hierarchical methods.** 

Answer: Flat clustering methods partition all of the data points into a fixed number of clusters and iteratively update the clusters. (Agglomerative) hierarchical methods begin with every point in its own cluster and then pick 2 two closest clusters to merge together and continue merging in the manner. 

```
平面聚类方法将所有数据点划分为固定数量的群，并迭代更新。 分层聚类方法从其自身群集中的每个点开始，然后选择2个两个最接近的群集以合并在一起并继续以这种方式合并
```

**b) Discuss the differences between soft and hard clustering.** 

Answer: Hard clustering assigns each point to at most one class, soft clustering assigns each point a portion (or probability) of membership to each cluster. 

```
硬聚类将每个点最多分配给一个类别，软聚类将每个点分配给每个聚类一部分概率
```

**c) Propose how you could adjust the K-means algorithm to give soft clusters instead of hard clusters.** 

Answer: We could assign soft membership scores based on the distance from a point to each of the clusters, the probability of point i belonging to cluster j would be given by 1−𝑑(𝑥𝑖,𝑐𝑗)Σ𝑑(𝑥𝑖,𝑐𝑘)|𝐶|𝑘=1 

```
我们可以基于从点到每个聚类的距离来分配评分
```

#### Q3: K-means Clustering Practice 

**Given the following set of 1 dimensional points: 1, 3, 6, 8, 20** 

**a) Run 2 iterations of k-means with Euclidean distance, k=2, and initial centres 1, 20.** 

Answer: 

Iteration 1 

Assignment: [1, 3, 6, 8], [20] 

Updated centres: [4.5], [20] 

Iteration 2 

Assignment: [1, 3, 6, 8], [20] 

Updated centres: [4.5], [20] 

**b) Run 2 iterations of k-means with Euclidean distance, k=3, and initial centres 1, 10, 20.** 

Answer: 

Iteration 1 

Assignment: [1, 3], [6, 8], [20] 

Updated centres: [2], [7], [20] 

Iteration 2 

Assignment: [1, 3], [6, 8], [20] 

Updated centres: [2], [7], [20] 

**c) Does k=2 or k=3 result in better clusters? Why?** 

Answer: k=3 seems to give a better clustering, because k=2 with only 2 clusters we have very uneven clusters, with one containing 4 points and the other only containing one point (due to the outlier). 

#### Q4: K-means Properties 

**For each of the following statements, decide whether they are true or false:** 

**a) K-means is guaranteed to converge to the globally best solution after a finite number of iterations.** 

Answer: False, it will converge but to a *locally* optimal solution. 

```
它会收敛，但会收敛到“局部”最优解
```

**b) The computational complexity of one iterations of k-means is 𝑂(𝑛𝑘2𝑑), where n is the number of data points and d is the dimension of the data points.** 

Answer: False, it should be 𝑂(𝑛𝑘𝑑). 

#### Q5: Purity 

**Suppose that you have run a clustering algorithm and found the following clusters: {𝑥1,𝑥2𝑥4 },{𝑥3,𝑥5}, and you additionally know that points are assigned to classes as follows: {𝑥1,𝑥2,},{𝑥3,𝑥4,𝑥5}** 

![image-20201110192111122](/Users/lizhuoran/Library/Application Support/typora-user-images/image-20201110192111122.png)

**a) Compute the purity between the assigned clusters and classes.** 

purity= 1/sum * max{交集} * max{交集}

Answer: 0.2 * 2 * 2 = 0.8 

**b) When used in this manner, is purity an intrinsic or extrinsic evaluation measure?** 

Answer: Extrinsic, because it uses external class labels to compare the clusters to. 

```
外在的，因为它使用外部类标签
```

#### Q6: Semantic Parsing 

**a) Give an example of an NLP task where using semantic parsing would be useful.** 

Answer: Any task where understanding semantic intent behind sentences is useful, e.g. machine translation, question answering, automatic code generation. 

```
理解句子背后的语义意图的任何任务都是有用的，例如机器翻译，问题解答，自动代码生成
```

#### Q7: Embeddings 

**a) Discuss the differences between distributional and distributed embeddings.** 

Answer: Distributional embeddings are computed from context statistics, distributed embeddings represent meaning by numerical vectors, rather than symbolic structures. 

```
分布嵌入是根据上下文统计数据计算得出的，分布嵌入通过数字矢量而不是符号结构来表示含义
```

#### Q8: Reference Resolution 

**For each of the following reference resolution models, briefly describe the model in terms of its inputs and key parameters.** 

**a) Mention-pair models** 

Answer: The model takes as input the embeddings of two mentions and outputs a binary decision, indicating whether or not they co-refer. 

```
该模型将两个提及的嵌入内容作为输入，并输出一个二进制决策，表明它们是否共同引用
```

**b) Mention ranking models** 

Answer: The model takes as input the embedding of a single mention, and outputs a softmax probability distribution over every previous mention. 

```
该模型将单个提及的嵌入作为输入，并输出每个先前提及的softmax概率分布。
```

## 10. syntax parsing and language models

#### Question 1: Parsing Methods 

**Discuss the differences between constituency parsing and dependency parsing. Suggest an example application where constituency parsing would be preferred, and one where dependency parsing would be preferred.** 

讨论成分句法分析（constituency parsing ）和依存句法分析（ dependency parsing）区别。

Answer: 

Dependency parsing can be more useful for several downstream tasks like Information Extraction or Question Answering. 

```
依赖关系解析对于一些下游任务（如信息提取或问题解答）可能会更加有用，例如，它很容易提取通常指示谓词之间语义关系的主语-动词-宾语三元组。尽管我们也可以从成分分析树中提取此信息，但它需要进行其他处理，而在依赖关系分析树中立即可用。
使用依赖解析器可能会证明是有利的另一种情况是使用自由字序语言时。顾名思义，这些语言不会对句子中的单词施加特定的顺序。由于基础语法的性质，在这种情况下，依赖项解析的性能要好于成分。
```

For example, it makes it easy to extract subject-verb-object triples that are often indicative of semantic relations between predicates. Although we could also extract this information from a constituency parse tree, it would require additional processing, while it’s immediately available in a dependency parse tree. 

Another situation where using a dependency parser might prove advantageous is when working with free word order languages. As their name implies, these languages don’t impose a specific order to the words in a sentence. Due to the nature of the underlying grammars, dependency parsing performs better than constituency in this kind of scenario. 

On the other hand, when we want to extract sub-phrases from the sentence, a constituency parser might be better. 

```
另一方面，当我们想从句子中提取子短语时，选区解析器可能会更好。
```

We can use both types of parse trees to extract features for a supervised machine learning model. The syntactical information they provide can be very useful for tasks like Semantic Role Labelling, Question Answering, and others. In general, we should analyse our situation and assess which type of parsing will best suit our needs. 

我们可以使用两种类型的解析树来提取监督型机器学习模型的特征。他们提供的语法信息对于诸如语义角色标签，问题回答等任务非常有用。通常，我们应该分析我们的情况并评估哪种类型的解析最适合我们的需求。

#### Question 2: Resolving Ambiguity 

**For both constituency and dependency parsing, there can be ambiguity in that there are multiple valid parses for a given sentences. Describe how you could automatically resolve such ambiguities.** 

Answer: 

We could estimate probabilities of parses using a training dataset (Probabilistic parsing). 

We could train a supervised machine learning model to predict which parse is most likely. 

使用训练集评估概率，也可以训练一个监督学习模型预测哪个好。

#### Question 3: Grammar Practice 

**Given the following grammar 𝐺=({𝐴,𝐵},{𝑎,𝑏},{(𝐴→𝑎𝐵),(𝐴→𝐵𝑎),(𝐴→𝑎𝑎),(𝐵→𝑏),(𝐵→𝐴𝑏)},𝐴} ,** 

**a) Is G a context free grammar?** 

Answer: Yes, all production rules have a single non-terminal on the left. 

```
是的，所有生产规则的左侧都有一个非终结符
```

**b) Find a valid derivation for the string aaabab** 

Answer: A -> aB -> aAb -> aBab -> aAbab -> aaabab 

**c) Is this grammar in Chomsky normal form?** 

Answer: No, there are rules with one terminal and rules with one non-terminal symbol on the right-hand side. 

```
不，在右侧有一个带有一个终止符的规则和一个带有一个非终止符的规则。
乔姆斯基范式的产生式要么右边是一个终结字符(单词)，要么是两个非终结符
```

#### Question 4: Markov Assumption 

**a) Describe the Markov assumption used in language models.** 

Answer: The nth-order Markov assumption states that the probability of a word occurring only depends on what the previous n words were. 

n阶马尔可夫假设指出，单词出现的概率仅取决于前n个单词是什么。

**b) Write down the chain rule decomposition of the probability 𝑝(𝑤1,𝑤2,𝑤3,𝑤4), given a second order Markov assumption.** 

Answer: 𝑝(𝑤1,𝑤2,𝑤3,𝑤4)=𝑝(𝑤1)∗𝑝(𝑤2|𝑤1)∗𝑝(𝑤3|𝑤1,𝑤2)∗𝑝(𝑤4|𝑤2,𝑤3) 

#### Question 5: Interpolation Methods 

**a) Describe what interpolation between language models is. Briefly discuss why this might be useful.** 

Answer: Interpolation refers to estimating probabilities as a weighted sum of higher-order and lower-order language models. This is useful because higher-order models are able to capture longer-term dependency between words however they are prone to overfitting, while lower-order models give reliable estimates of rare words, but do not take into account order. By interpolating, a model can be made which has the strengths of both. 

**b) In general, describe how the use of p_continuation in Kneser-Ney smoothing affects its probability estimates compared to absolute discounting.** 

Answer: p_continuation calculates how likely the word is to appear as a continuation of the previous word, which is not dependent on the number of times the word appears, but rather the number of times it appears as a continuation. This means that if one word is frequent but only ever appears next to a small number of other words (e.g. San Francisco) Kneser-Ney will assign it a lower probability than absolute discounting. 

#### Question 6: Language Model Practice 

**Given the following corpus of text consisting of 3 sentences:** 

**Can the cat catch** 

**Can catch a cat** 

**The cat can catch a can** 

**a) Compute the probability P(cat|can) using absolute discounting.** 

![image-20201112111814130](/Users/lizhuoran/Library/Application Support/typora-user-images/image-20201112111814130.png)

**b) computer the probability P(cat|can) using Kneser-Ney smoothing.**

![image-20201112111845521](/Users/lizhuoran/Library/Application Support/typora-user-images/image-20201112111845521.png)

**c) Compute the probability P(cat|can) using stupid back-off.** 

Answer: 0.4 * 3/14 = 0.0857 

**d) Which one of these methods do you think gives a more accurate estimation of the probability p(cat|can)?** 

Answer: For this example, Kneser-Ney would likely give a better estimate since the word cat only appears after 2 unique words. 

**e) Compute the perplexity of the sequence *cat, catch, can* using a stupid back-off model** 

![image-20201112111914795](/Users/lizhuoran/Library/Application Support/typora-user-images/image-20201112111914795.png)

## 11. Dependency parsing and NLP structures

####  Question 1: Well-formedness 

**For each of the following dependency graphs, state whether they are well-formed, and if not why not.** 

![image-20201112111941015](/Users/lizhuoran/Library/Application Support/typora-user-images/image-20201112111941015.png) 

Answer: Only b) is well-formed. a) is non-projective, c) is not connected, and d) has a word with multiple heads. 

#### Question 2: Transition requirements 

**For each of the transitions in Nivre’s arc-eager parsing algorithm, explain what the pre-condition for this transition is and why it is necessary (what would go wrong if the transition was applied when the pre-condition is False?)** 

Answer: 

Left-arc: There needs to be a rule which says that the word at the front of the input can be the head of the word on the top of the stack, and the word on the top of the stack can’t already have a head. This pre-condition needs to be true or else the transition would make the graph multi-headed. 

Right-arc: Same as left-arc, but reversed. 

Reduce: The word on the top of the stack must already have a head. If this condition is false, then applying reduce may make the graph not connected. 

#### Question 3: Nivre’s arc-eager parser 

**Consider the following sentence:** 

**the cat jumped on the fence** 

**[DET, NOUN, VERB, PREP, DET, NOUN]** 

**Run Nivre’s arc-eager parsing algorithm on this sentence with the following set of rules:** 

**Root → Verb,** 

**Verb → Noun,** 

**Noun → Det,** 

**Det → Prep** 

**And this prioritization of transitions:** 

**LA > RA > R > S** 

Answer: 

[ROOT], the cat jumped on the fence, {} 

[the, ROOT], cat jumped on the fence, {} (S) 

[ROOT], cat jumped on the fence, {cat → the} (LA) 

[cat, ROOT], jumped on the fence, {cat → the} (S) 

[ROOT], jumped on the fence, {cat → the, jumped → cat} (LA) 

[jumped, ROOT], on the fence, {cat → the, jumped → cat, ROOT → jumped} (RA) 

[ROOT], on the fence, {cat → the, jumped → cat, ROOT → jumped} (R) 

[on, ROOT], the fence, {cat → the, jumped → cat, ROOT → jumped} (S) 

[ROOT], the fence, {cat → the, jumped → cat, ROOT → jumped, the → on} (LA) 

[the, ROOT], fence, {cat → the, jumped → cat, ROOT → jumped, the → on} (s) 

[ROOT], fence, {cat → the, jumped → cat, ROOT → jumped, the → on, fence → the} (LA) 

[fence, ROOT], ,{cat → the, jumped → cat, ROOT → jumped, the → on, fence → the} (s) 

PARSING FAIL 

#### Question 4: Task structures 

**For each of the following tasks, which data structures would you use to represent the inputs and outputs?** 

**a) Text classification.** 

Answer: Sequence input, vector output. 

**b) Natural language translation.** 

Answer: Sequence input, sequence output. 

**c) Automatic code generation.** 

Answer: Sequence input, tree output. 

**d) Tweet classification.** 

Answer: Graph (of sequences) input, vector output. 

**e) Question-Answering (Given a paragraph of a text, and a natural language question asking something about this text, output the answer to that question).** 

Answer: Graph two sequences as input, sequence output. 

#### Question 5: Task Pipeline 

**For the following tasks, create rough pipelines for solving them. Mention what kind of data you would need to solve the task and which specific architecture you would use (there can be more than one).** 

**A) Convert instructions in plain English about a website UI into HTML + CSS code** 

![image-20201111221032845](/Users/lizhuoran/Library/Application Support/typora-user-images/image-20201111221032845.png)

**B) Construct a generative chatbot that can hold a conversation (it shouldn’t just reply to the current input, but hold some context from previous input and output too)** 

![image-20201111221013092](/Users/lizhuoran/Library/Application Support/typora-user-images/image-20201111221013092.png)

**C) Predict future stock market prices, using information sources such as news articles and social media.** 

![image-20201111221022943](/Users/lizhuoran/Library/Application Support/typora-user-images/image-20201111221022943.png)