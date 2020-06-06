from flask import Flask
from flask_cors import CORS
from flask import render_template
import os

app = Flask(__name__)


@app.route('/')
def hello_world():
    image1 = []
    image2 = []
    for file in os.listdir("./static/images"):
        if file.endswith(".png"):
            image1.append(os.path.join("./static/images",file))
    for file in os.listdir("./static/images"):
        if file.endswith(".png"):
            image2.append(os.path.join("./static/edged_images",file))
    image1 =sorted(image1,key = lambda i:int(i.split("/")[-1].split(".")[0]))
    print(image1)
    image2 = sorted(image2, key=lambda i: int(i.split("/")[-1].split(".")[0]))
    return render_template('index.html',image=image2,edged_image=image1)

def open_flask():
    CORS(app, supports_credentials=True)
    app.jinja_env.auto_reload = True
    app.run(host='0.0.0.0', port=5000, debug=False)


if __name__ == '__main__':
    open_flask()

