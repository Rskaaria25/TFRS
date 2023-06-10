import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
import tensorflow as tf

# Load the dataset
data = pd.read_csv('food_test.csv')

# Preprocessing the dataset
data['Tipe'] = data['Tipe 1'] + ',' + data['Tipe 2'] + ',' + data['Tipe 3']
data['Tipe'] = data['Tipe'].str.lower()

# Vectorizing pre-processed food type plots using TF-IDF
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(data['Tipe'])

# Finding cosine similarity between vectors
similarity_matrix = cosine_similarity(tfidf_matrix)

# Define the FoodRecommenderModel
class FoodRecommenderModel(tf.keras.Model):
    def __init__(self, num_food_names, num_food_types, embedding_dim=32, hidden_units=[64, 32]):
        super(FoodRecommenderModel, self).__init__()
        self.food_name_embedding = Embedding(num_food_names, embedding_dim)
        self.food_type_embedding = Embedding(num_food_types, embedding_dim)
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(units, activation='relu') for units in hidden_units
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(units, activation='relu') for units in reversed(hidden_units[:-1])
        ])
        self.output_layer = tf.keras.layers.Dense(num_food_names, activation='softmax')

    def call(self, inputs):
        food_names = inputs['food_names']
        food_types = inputs['food_types']
        
        food_name_embeddings = self.food_name_embedding(food_names)
        food_type_embeddings = self.food_type_embedding(food_types)
        concatenated_embeddings = tf.keras.layers.concatenate([food_name_embeddings, food_type_embeddings], axis=1)
        encoded = self.encoder(concatenated_embeddings)
        decoded = self.decoder(encoded)
        logits = self.output_layer(decoded)
        return logits

# Define the function to get recommendations
def get_recommendations_by_types(food_types, similarity_matrix, k=10):
    food_indices = []
    for food_type in food_types:
        indices = data[data['Tipe'].str.contains(food_type)].index
        food_indices.extend(indices)
    food_indices = list(set(food_indices))
    
    similarity_scores = list(enumerate(similarity_matrix[food_indices].sum(axis=0)))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_k_food_indices = [i for i, _ in similarity_scores[1:k+1]]
    top_k_food_names = data.loc[top_k_food_indices, 'Nama'].values
    return top_k_food_names

# Create the StringLookup layers for food names and types
food_name_vocab = StringLookup()
food_name_vocab.adapt(data['Nama'].values)
food_type_vocab = StringLookup()
food_type_vocab.adapt(data['Tipe'].values)

# Convert the input data to TensorFlow Dataset
def convert_to_dataset(names, types):
    names = food_name_vocab(names)
    types = food_type_vocab(types)
    return tf.data.Dataset.from_tensor_slices((names, types)).batch(32)

# Convert the dataset to TensorFlow Dataset
train_data = convert_to_dataset(data['Nama'], data['Tipe'])

# Create an instance of the FoodRecommenderModel
model = FoodRecommenderModel(
    num_food_names=len(food_name_vocab.get_vocabulary()),
    num_food_types=len(food_type_vocab.get_vocabulary())
)

# Define the loss function
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

# Define the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Define the metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')

@tf.function
def train_step(inputs):
    food_names, food_types = inputs
    with tf.GradientTape() as tape:
        logits = model({"food_names": food_names, "food_types": food_types})
        loss_value = loss_object(food_names, logits)
    gradients = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss_value)

# Train the model
epochs = 10
for epoch in range(epochs):
    for inputs in train_data:
        train_step(inputs)
    print('Epoch {}/{} - Loss: {}'.format(epoch + 1, epochs, train_loss.result()))

# Get recommendations for specific food types
food_types = ['jajanan']
recommendations = get_recommendations_by_types(food_types, similarity_matrix, k=10)
print('Recommendations for {}:'.format(', '.join(food_types)))
for recommendation in recommendations:
    print('- {}'.format(recommendation))