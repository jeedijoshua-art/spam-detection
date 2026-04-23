import os
import json
import base64
import google.oauth2.credentials
from flask import Flask, render_template, request, jsonify, redirect, session
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

from model import predict_email

app = Flask(__name__)
# Generate a random secret key so sessions are invalidated on server restart
app.secret_key = os.urandom(24)

# Gmail config
try:
    GOOGLE_CLIENT_CONFIG = json.loads(os.environ.get("GOOGLE_CLIENT_SECRET", "{}"))
except Exception as e:
    print("ERROR LOADING GOOGLE CLIENT SECRET:", e)
    GOOGLE_CLIENT_CONFIG = {}
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']


# ==========================================
# HELPER FUNCTIONS
# ==========================================

@app.before_request
def clear_session_on_start():
    if 'initialized' not in session:
        session.clear()
        session['initialized'] = True

def credentials_to_dict(credentials):
    """Helper function to store credentials in Flask session safely."""
    return {
        'token': credentials.token,
        'refresh_token': credentials.refresh_token,
        'token_uri': credentials.token_uri,
        'client_id': credentials.client_id,
        'client_secret': credentials.client_secret,
        'scopes': credentials.scopes
    }

def get_message_body(payload):
    """Recursively extract plain text body from the payload."""
    if 'data' in payload.get('body', {}):
        return base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8', errors='replace')
    if 'parts' in payload:
        for part in payload.get('parts', []):
            if part.get('mimeType') == 'text/plain':
                if 'data' in part.get('body', {}):
                    return base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='replace')
        # Check sub-parts
        for part in payload.get('parts', []):
            if 'parts' in part:
                res = get_message_body(part)
                if res: return res
    return ""


# ==========================================
# EXISTING ROUTES (UI & ML)
# ==========================================

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

@app.route("/", methods=["GET"])
def home():
    if session.get("logged_in"):
        return redirect("/dashboard")
    return render_template("login.html")

@app.route("/dashboard")
def dashboard():
    if not session.get("logged_in"):
        return redirect("/")
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """Takes in raw text and returns a spam probability"""
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400
    
    text = data["text"]

    if not text.strip():
        return jsonify({
            "prediction": 0,
            "spam_probability": 0.0,
            "safe_probability": 1.0,
            "text_length": 0,
            "word_count": 0
        })
        
    result = predict_email(text)
    result["text_length"] = len(text)
    result["word_count"] = len(text.split())
    
    return jsonify(result)


# ==========================================
# GMAIL OAUTH & API ROUTES
# ==========================================

@app.route("/login")
def login():
    """Initiates the OAuth2 flow by redirecting to Google."""
    if not GOOGLE_CLIENT_CONFIG or "web" not in GOOGLE_CLIENT_CONFIG:
        return "Google OAuth not configured properly", 500

    # Ensure fresh execution by not reusing previous state or credentials
    session.pop('credentials', None)
    session.pop('state', None)

    flow = Flow.from_client_config(
        GOOGLE_CLIENT_CONFIG,
        scopes=SCOPES,
        redirect_uri=request.url_root + "callback"
    )

    auth_url, state = flow.authorization_url(
        prompt='consent',
        access_type='offline'
    )

    # Store OAuth context in session
    session['state'] = state
    session['code_verifier'] = flow.code_verifier

    return redirect(auth_url)


@app.route("/callback")
def callback():
    """Handles the callback from Google, completing the OAuth flow."""
    if not GOOGLE_CLIENT_CONFIG or "web" not in GOOGLE_CLIENT_CONFIG:
        return "Google OAuth not configured properly", 500

    flow = Flow.from_client_config(
        GOOGLE_CLIENT_CONFIG,
        scopes=SCOPES,
        state=session['state'],
        redirect_uri=request.url_root + "callback"
    )

    flow.code_verifier = session.get('code_verifier')
    flow.fetch_token(authorization_response=request.url)

    # Store credentials in session dict (avoids massive session cookie bloat)
    session['credentials'] = credentials_to_dict(flow.credentials)
    session['logged_in'] = True

    # Redirect directly back to the SPA homepage
    return redirect("/dashboard")


@app.route("/api/emails", methods=["GET"])
def get_emails():
    """Fetches ONLY the id and snippet (preview) for fast initial loading."""
    if 'credentials' not in session:
        return jsonify({"error": "Not authenticated", "emails": []}), 401
    
    try:
        creds = google.oauth2.credentials.Credentials(**session['credentials'])
        service = build('gmail', 'v1', credentials=creds)
        
        emails = []
        
        results = service.users().messages().list(
            userId='me',
            maxResults=10
        ).execute()
        
        messages = results.get('messages', [])
        
        for msg in messages:
            msg_id = msg['id']
            
            # Use format='minimal' to get ONLY snippet and ID
            # This avoids fetching the expensive headers and body payload.
            msg_data = service.users().messages().get(
                userId='me',
                id=msg_id,
                format='minimal'
            ).execute()

            snippet = msg_data.get('snippet', '')
            
            emails.append({
                "id": msg_id,
                "preview": snippet
            })
            
        return jsonify({"emails": emails})
        
    except Exception as e:
        print(f"Error fetching emails from API: {e}")
        return jsonify({"error": "Failed to fetch emails via Gmail API", "emails": []}), 500

@app.route("/api/email/<id>", methods=["GET"])
def get_single_email(id):
    """Fetches full content of a single email, utilizing cache."""
    if 'credentials' not in session:
        return jsonify({"error": "Not authenticated"}), 401
        
    try:
        if 'emails_cache' not in session:
            session['emails_cache'] = {}
            
        # Check if already fetched
        cache = session['emails_cache']
        if id in cache and 'body' in cache[id]:
            return jsonify(cache[id])
            
        creds = google.oauth2.credentials.Credentials(**session['credentials'])
        service = build('gmail', 'v1', credentials=creds)
        
        msg_data = service.users().messages().get(
            userId='me',
            id=id,
            format='full'
        ).execute()
        
        headers = msg_data.get('payload', {}).get('headers', [])
        subject = "No Subject"
        sender = "Unknown Sender"
        date = "12:00 PM"
        
        for h in headers:
            name = h.get('name', '').lower()
            if name == 'subject':
                subject = h.get('value')
            elif name == 'from':
                sender = h.get('value')
            elif name == 'date':
                date = h.get('value')[:25]
                
        body_text = get_message_body(msg_data.get('payload', {}))
        if not body_text.strip():
            body_text = msg_data.get('snippet', '')

        email_data = {
            "id": id,
            "subject": subject,
            "from": sender,
            "sender": sender,
            "time": date,
            "preview": msg_data.get('snippet', ''),
            "body": body_text
        }
        
        # Save to cache
        cache[id] = email_data
        session.modified = True
        
        return jsonify(email_data)
        
    except Exception as e:
        print(f"Error fetching single email from API: {e}")
        return jsonify({"error": "Failed to fetch email"}), 500


# ==========================================
# RUN EXECUTOR
# ==========================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))