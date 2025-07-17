from http.server import BaseHTTPRequestHandler
import joblib
import json
from urllib.parse import parse_qs
import io

model = joblib.load('./spam_detector_model.pkl')

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        data = json.loads(body)
        text = data.get('text')
        result = model.predict([text])[0]
        response = {
            "spam": bool(result)
        }
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(bytes(json.dumps(response), "utf-8"))
        return
