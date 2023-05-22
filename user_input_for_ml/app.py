from flask import Flask, render_template
from flask import jsonify


app = Flask(__name__)
# webcode = open('webcode.html').read() - not needed

@app.route('/')
def webprint():
    return render_template('inp.html') 

if __name__ == '__main__':
    app.run(debug=True)