from flask import Flask, request, render_template
from recommendation import get_recommendations_by_types

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        food_types = request.form.get('food_types')
        food_types = food_types.lower().split(',')
        recommendations = get_recommendations_by_types(food_types, k=10)
        return render_template('index.html', recommendations=recommendations)
    return render_template('index.html')

if __name__ == '__main__':
    app.run()

