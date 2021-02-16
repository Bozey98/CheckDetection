from flask import Flask, request
import MLModule
app = Flask(__name__)

@app.route('/get_check_data')
def show_user_profile():
    url = request.args.get("url")

    return MLModule.get_info_from_check(url)