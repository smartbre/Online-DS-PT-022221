# Natural Language Processing
![](static/oprah-everyone.png)

Today we will be doing some EDA with the nltk library, and fitting machine learning models using text as predictors.


```python
# Base Libraries
import pandas as pd
import numpy as np
import string

# NLP
import nltk
from nltk.stem import PorterStemmer
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Visualization
import matplotlib.pyplot as plt

# Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
```

In the cell below, we import the policy proposal by 2020 Democratic Presidential Candidates Bernie Sanders and Elizabeth Warren.


```python
df = pd.read_csv('data/2020_policies_feb_24.csv')
df.head()
```

**We need to do some processing to make this text usable.** 

In the cell below, define a function called `prepocessing` that receives a single parameter called `text`.

<u><b>This function should:</b></u>
1. Lower the text so all letters are the same case
2. Use nltk's `word_tokenize` function to convert the string into a list of tokens.
3. Remove stop words from the data using nltk's english stopwords corpus.
4. Use nltk's `PortStemmer` to stem the text data
5. Remove punctuation from the data 
    - *(You can use the [string](https://www.journaldev.com/23788/python-string-module) library for this)*
6. Convert the list of tokens into a string
7. Remove opening and trailing spaces, and replace all double spaces with a single space.
8. Return the results.


```python
## The below code may need to be run in for
## you to use the nltk PortStemmer
## and word_tokenize

# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
```


```python
def preprocessing(text):
    # Initialize a PortStemmer object
    # YOUR CODE HERE
    
    # Initialize nltk's stopwords
    # YOUR CODE HERE
    
    # Lower the text
    # YOUR CODE HERE  
    
    # Remove punctuation
    # YOUR CODE HERE
    
    # Tokenize the no-punctuation text
    # YOUR CODE HERE    
    
    # Remove stop words
    # YOUR CODE HERE  
    
    # Convert the tokens into their stem
    # YOUR CODE HERE  
    
    # Convert the list of words back into
    # a string by joining each word with a space
    # YOUR CODE HERE   
    
    # Remove double spaces
    # YOUR CODE HERE   
    
    # Remove opening and trailing spaces
    # YOUR CODE HERE   
    
    # Return the cleaned text data
    # YOUR CODE HERE
```

Examine the output for the following cell. 
- Was your code successful? 
- Are there words in the output that should be added to our list of stopwords?
- Should we remove numbers?


```python
preprocessing(df.policy[0])[:500]
```

**Let's apply our preprocessing to every policy.**


```python
df.policy = df.policy.apply(preprocessing)

print(df.policy[:3])
```

Now, we can explore our text data.

In the cell below define a function called `average_word_length` that receives a single parameter called `text`, and outputs the average word length.

<u><b>This function should:</b></u>
1. Split the text into a list of tokens
2. Find the length of every word in the list
3. Sum the word lengths and divide by the number of words in the list of tokens.
4. Return the result.


```python
def average_word_length(text):
    # Split the text
    # YOUR CODE HERE
    
    # Calculate the sum of each word length
    # and divide by the total number of words
    # YOUR CODE HERE

    # Return the calculation
    # YOUR CODE HERE
```

Now, we apply our function to every policy and add the output as column.


```python
df['average_word_length'] = df.policy.apply(average_word_length)
```

Sweet let's take a look at the documents with the highest average word length.


```python
df.sort_values(by='average_word_length', ascending=False).head()
```

An average measurement can be a bit misleading. 

Let's also write a function that finds the word count for a given document.

In the cell below, define a function called `word_count` that receives a single `text` parameter.

<u><b>This function should:</b></u>
1. Split the text data
2. Return the length of the array.


```python
def word_count(text):
    # Split the text
    # YOUR CODE HERE
    
    # Find the number of words
    # in the split text
    # YOUR CODE HERE
```

Nice. Now we apply the function to our entire dataset, and save the output as a column


```python
df['word_count'] = df.policy.apply(word_count)

df.sort_values(by='average_word_length', ascending=False).head()
```

Interesting. Let's take a look at the distribution for the `word_count` column.


```python
warren_df = df[df.candidate=='warren']
sanders_df = df[df.candidate=='sanders']

plt.figure(figsize=(15,6))
plt.hist(warren_df.word_count, alpha=.6, label='Warren')
plt.hist(sanders_df.word_count, alpha=.6, label='Sanders')
plt.legend()
plt.show()
```

It looks like the average length of a policy is about 1,000 words.

Let's print the mean and median for the `word_count` column.


```python
print('Mean Word Count: ',df.word_count.mean())
print('Median Word Count: ',df.word_count.median())
```

*There are some outliers in this data that in a full data science project would would be worth exploring!*

Let's find out what the most frequent words are for each candidate.

First, we use list comprehension to create a list of token-lists.


```python
token_warren= [word_tokenize(policy) for policy in warren_df.policy] 
```

Next, we want to create a bag of words. AKA a single list containing every token.


```python
warren_bow = []
for doc in token_warren:
    warren_bow.extend([word.lower() for word in doc])
```

Now, we use the `FreqDist` object to find the 10 most frequent words.


```python
fd_warren = FreqDist(warren_bow)
print(fd_warren.most_common(10))
```

Are there any words here that should be added to our list of stopwords?

*In the cell below* define a function called `word_frequency` that receives a series of documents, and outputs a fitted FreqDist object.

<u><b>This function should be</b></u> a generalized version of the code we just wrote, only instead of printing out the most frequent words, the function should return an Initialized `FreqDist` object.


```python
def word_frequency(documents):
    # Tokenize each of the documents
    # YOUR CODE HERE
    
    # Create an empty list
    # YOUR CODE HERE
    
    # Loop over each tokenized document
    # YOUR CODE HERE
    
        # Add all of the tokens to the empty list
        # YOUR CODE HERE
        
    # Initialize a FreqDist object
    # YOUR CODE HERE
    
    # Return the FreqDist object
    # YOUR CODE HERE
```

Now, we can feed all of sanders policies into our `word_frequency` functions and receive a fitted `FreqDist` object


```python
fd_sanders = word_frequency(sanders_df.policy)
fd_sanders.most_common(10)
```

`FreqDist` objects come with a handy `.plot` method :)


```python
fd_sanders.plot(10, title='Bernie Sanders - Most Common Words');
```


```python
fd_warren.plot(10, title='Elizabeth Warren - Most Common Words');
```

## Classification

It looks like there are some more words we could probably add as stopwords. This is a pretty common iteration that is seen in Natural Language projects. It's pretty typical to  drop initial stopwords, evaluate the frequency distribution of the cleaned text, fit models to the text, and evaluate what words are most important/most common. Depending on your modeling goal, it can oftentimes take several iterations to ensure that your data does not contain obvious indicators for your target. For instance, in this data it would probably be a good idea to remove the names of the candidate from the text if we really wanted to see how a model differentiates between the two candidates based on the content of their policies.

Let's see how many policies for each candidate mention the candidate by name.


```python
warren_perc = warren_df[warren_df.policy.str.contains('warren')].shape[0]/warren_df.shape[0]
sanders_perc = sanders_df[sanders_df.policy.str.contains('berni')].shape[0]/sanders_df.shape[0]
string_template = '{} percent: {:.2%}'
print(string_template.format('Sanders', sanders_perc))
print(string_template.format('Warren', warren_perc))
```

Let's see if we can remove references to the candidates themselves.


```python
# Helper function to remove specific words from the dataset
def remove_new_stopwords(text, words):
    new_text = str(text)
    for word in words:
        new_text = new_text.replace(word, '')
    return new_text

# Remove the names of the candidates from the policies
warren_df = warren_df.assign(policy = warren_df.policy.apply(lambda x: remove_new_stopwords(x, ['warren', 'elizabeth'])))
sanders_df = sanders_df.assign(policy = sanders_df.policy.apply(lambda x: remove_new_stopwords(x, ['berni', 'sander'])))
```

The percentages should now be at 0%


```python
warren_perc = warren_df[warren_df.policy.str.contains('warren')].shape[0]/warren_df.shape[0]
sanders_perc = sanders_df[sanders_df.policy.str.contains('berni')].shape[0]/sanders_df.shape[0]
string_template = '{} percent: {:.2%}'
print(string_template.format('Sanders', sanders_perc))
print(string_template.format('Warren', warren_perc))
```

Let's concatenate these two tables together and put together a target columm for modeling.

In this case, we will create a target column that indicates the name of the candidate.


```python
from sklearn.preprocessing import LabelEncoder

# Concatenate the two datasets
modeling_data = pd.concat([warren_df, sanders_df])

# Fit a label encode to the column 
# indicating the name of the candidate
target_encoder = LabelEncoder()

# Transform to candidate column to an array of [0,1]
target = target_encoder.fit_transform(modeling_data.candidate)
```

Now that we have our target column, let's create a train test split of the data.


```python
from sklearn.model_selection import train_test_split
                    
X_train, X_test, y_train, y_test = train_test_split(modeling_data[['policy']], # Isolating the policy column
                                                    target, random_state=2021)
```

Good, now let's create some pipelines for different tokenization strategies.

In the cell below, import `CountVectorizer` and `TfidfVectorizer` from sklearn


```python
# YOUR CODE HERE
```

In the cell below, initialize two vectorizers using the following variable names
1. `count`
2. `tfidf`

> Small note: Please take a second to notice that both of these vectorizors have a `stop_words` hyperparameter! Using this hyperparameter, we can pass in a list of stopwords to the the vectorizer and the vecotrizer will vectorize *and* remove stopwords all at once. This is not important to us, given that we have already removed stopwords, but it is important to recognize that this can dramatically streamline your preprocessing with text data.


```python
# YOUR CODE HERE
```

Now we will create a dictionary containing different pipelines for each of our vectorization strategies.


```python
from sklearn.pipeline import make_pipeline
random_state = 2021

models = {'lr_count': make_pipeline(count, LogisticRegression(random_state=random_state)),
          'dt_count': make_pipeline(count, DecisionTreeClassifier(random_state=random_state)),
          'rf_count': make_pipeline(count, RandomForestClassifier(random_state=random_state)),
          'lr_tfidf': make_pipeline(tfidf, LogisticRegression(random_state=random_state)),
          'dt_tfidf': make_pipeline(tfidf, DecisionTreeClassifier(random_state=random_state)),
          'rf_tfidf': make_pipeline(tfidf, RandomForestClassifier(random_state=random_state))}
```

And then we can run our models!


```python
from sklearn.model_selection import cross_val_score

baseline_scores = {}

for model in models:
    score = cross_val_score(models[model], X_train.iloc[:,0], y_train, scoring='f1')
    baseline_scores[model] = score.mean()
    
baseline_scores
```

It looks like our best performing model is the RandomForest model using tfidf vectorization. Let's see how this modeling is doing.

To do that, we will pull the model out of our models dictionary.


```python
rf_pipeline = models['rf_tfidf']
```

Now fit the model to the training data!


```python
# YOUR CODE HERE
```

Let's inspect the confusion matrix for our two sets of data.


```python
from sklearn.metrics import plot_confusion_matrix

fig, ax = plt.subplots(1,2, figsize=(15,6))
labels = target_encoder.inverse_transform([0,1])

plot_confusion_matrix(rf_pipeline, X_train.iloc[:,0], y_train, ax=ax[0], display_labels=labels)
ax[0].set_title('Training Data')

plot_confusion_matrix(rf_pipeline, X_test.iloc[:,0], y_test, ax=ax[1], display_labels=labels)
ax[1].set_title('Testing Data');
```

100% accuracy on our training data and then it looks like we're predicting Bernie Sanders with about a 50% recall score.

Let's inspect what features the model is using for prediction by using `permutation_importance`. Because we used a pipeline, we will need pull out the individual objects


```python
# the fit tfidf vectorizer
transformer = rf_model.steps[0][-1]
# the fit random forest model
rf_model = rf_model.steps[-1][-1]
```

Next we will import `permutation_importance` from sklearn


```python
from sklearn.inspection import permutation_importance
```

Transform our our testing data with the fit tfidf vectorizer:


```python
X_inspect = transformer.transform(X_test.iloc[:,0]).toarray()
```

And pass our model, the transformed data, and the target into `permutation_importance`.


```python
importance = permutation_importance(rf_model, X_inspect, y_test, random_state=2021, scoring='f1')
```

The cell above will take a moment to run, so while it runs, we may as well talk about what `permutation_importance` is doing. 

Researchers have found that the feature importances given from `.get_feature_importance` returns bias information that does not accurately reflect how how much predictive information a feature provides to the model. You can read more about this [here](https://explained.ai/rf-importance/#:~:text=Permutation%20importance,-Breiman%20and%20Cutler&text=Permute%20the%20column%20values%20of,caused%20by%20permuting%20the%20column.). The recommended solution to this problem is to use `permutation importance`. **Permutation Importance** loops over every feature in your dataset and for each iteration will randomly shuffle the data in a feature's column. By doing so, the relationship between the target and the feature is severed. Once this has been done for every feature, features are given a weight based on how poorly the model did when the feature's data was scrambled. If the model did a lot worse, that suggests that model really needs that feature, thus it is importance. If the model did exactly the same, then the feature is marked as unimportant.

Let's take a look at what words were considered most importance by our Random Forest model.


```python
# Zip the names of the features 
# with the features permutation importance
importance_weights = list(zip(transformer.get_feature_names(), importance['importances_mean']))

# Sort the weights in descending order
sorted(importance_weights, key=lambda x: x[1], reverse=True)[:100]
```

Looking at the feature importances above, it looks like after the top 73 features, the remaining features are not considered importance. Let's drop them and see how the model does. We will also drop features where the word is a number, as this seems sort of nonsensical.


```python
# Sort the features in descending order based on their
# permutation importance
top_features = sorted(importance_weights, key=lambda x: x[1], reverse=True)[:73]

# Isolate the name of the feature
top_features = [x[0] for x in top_features if not x[0].isdigit()]
```

Cool cool, now we will... 
1. Transform the training and testing data with out tfidf vectorizer
2. Set the feature names as the column for the transformed data
3. Isolate the features with the most predictive power.


```python
# Transform the training and testing data with out tfidf vectorizer
X_train_transformed = pd.DataFrame(tfidf.transform(X_train.iloc[:,0]).toarray())
X_test_transformed = pd.DataFrame(tfidf.transform(X_test.iloc[:,0]).toarray())

# Set the feature names as the column for the transformed data
X_train_transformed.columns = tfidf.get_feature_names()
X_test_transformed.columns = tfidf.get_feature_names()

# Isolate the features with the most predictive power.
X_train_top_features = X_train_transformed[top_features]
X_test_top_features = X_test_transformed[top_features]
```

And then we fit the model to our filtered training data:


```python
rf_model.fit(X_train_top_features, y_train)
```

And now let's plot confusion matrices for both data splits:


```python
fig, ax = plt.subplots(1,2, figsize=(15,6))

plot_confusion_matrix(estimator, X_train_top_features, y_train, ax=ax[0])
plot_confusion_matrix(estimator, X_test_top_features, y_test, ax=ax[1])
ax[0].set_title('Training')
ax[1].set_title('Testing');
```

Much better performance! The last thing to note about feature importance is that it tells us absolutely nothing about the *relationship* of the feature, other than "it is informative". For example, from this modeling process we have learned that the word "Moratorium" contains predictive information, but we have no idea whether or not "Moratorium" is predictive of Warren or if it predictive of Sanders. 

A first step in analyzing our feature importances is to visualize their relationship with the target column

Let's visualize the percentage of policies that contain the top 25 words for each candidate.


```python
fig, axes = plt.subplots(5,5, figsize=(20,20))

for idx in range(25):
    word = top_features[idx]
    row, col = idx//5, idx%5
    ax = axes[row, col]
    w_count = warren_df[warren_df.policy.str.contains(word)].shape[0]/warren_df.shape[0]
    s_count = sanders_df[sanders_df.policy.str.contains(word)].shape[0]/sanders_df.shape[0]
    ax.bar(['Warren', 'Sanders'], [w_count, s_count], color=['#b7e4d0','#0370cc'])
    ax.set_title(word.title(), fontsize=20)
fig.tight_layout()
```

If we look at the above, we can see that `Detail` is in 100% of the Sanders Policies. This is because every Sanders policy began with bullet points, and a title containing the word "Detail". For a future iteration, if we are wanting the feature weights of our model to be more informative, we might consider adding the word "detail" to our list of stopwords. 
