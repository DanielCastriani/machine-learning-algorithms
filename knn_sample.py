import pandas as pd
import seaborn as sns
from utils import shuffle, train_test_split, min_max_normalize, label_encoder, confusion_matrix, accuracy 
from neighbors import KNNClassifier

iris = pd.read_csv('./data/Iris.csv')

iris['Species'], labels = label_encoder(iris['Species'].values.tolist())

train, test = train_test_split(iris, train_size=0.7)

x_train = train.drop('Species', axis=1)
x_test = test.drop('Species', axis=1)

y_train = train['Species']
y_test = test['Species']


# if use the entire dataset, to generate the min and max values, will be 'cheating on the game'. 
# You need to normalize test using train min and max. This is also valid for new entries in production
x_train, minmax = min_max_normalize(x_train)
x_test, _ = min_max_normalize(x_test)

x_train.describe()

knn = KNNClassifier(k=5)
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)

acc = accuracy(y_test, y_pred) * 100
conf = confusion_matrix(y_test, y_pred)
print('acc %.2f' % acc)
sns.heatmap(conf, annot=True)
