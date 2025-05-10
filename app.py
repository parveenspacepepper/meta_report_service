import requests
import pandas as pd
import os
import tempfile
from datetime import datetime, timedelta
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import smtplib
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import logging
from flask import Flask, jsonify
import traceback
import pytz

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize Flask app
app = Flask(__name__)

# Set timezone to IST
IST = pytz.timezone('Asia/Kolkata')

def get_current_timestamp():
    now = datetime.now(IST)
    today = now.strftime('%Y-%m-%d')
    timestamp_str = now.strftime('%Y-%m-%d / %I:%M %p IST').lstrip('0')
    return today, timestamp_str


# Load environment variables
load_dotenv()
ACCESS_TOKEN = os.getenv('META_API_KEY')
ACCOUNT_ID = os.getenv('META_AD_ACCOUNT_ID')
EMAIL_SENDER = os.getenv('EMAIL_SENDER', '')
EMAIL_RECIPIENTS = os.getenv('EMAIL_RECIPIENTS', '').split(',')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')
ENV = os.getenv('ENV', 'development')
REPORT_DIR = os.getenv('REPORT_DIR', os.path.join(tempfile.gettempdir(), 'reports'))

# Ensure report directory exists
os.makedirs(REPORT_DIR, exist_ok=True)

# Validate environment variables
required_env_vars = {
    'META_API_KEY': ACCESS_TOKEN,
    'META_AD_ACCOUNT_ID': ACCOUNT_ID,
    'EMAIL_SENDER': EMAIL_SENDER,
    'EMAIL_RECIPIENTS': EMAIL_RECIPIENTS,
    'EMAIL_PASSWORD': EMAIL_PASSWORD
}
for var_name, var_value in required_env_vars.items():
    if not var_value or (var_name == 'EMAIL_RECIPIENTS' and not var_value[0]):
        logging.error(f"Environment variable {var_name} is missing or invalid.")
        raise ValueError(f"Environment variable {var_name} is missing or invalid.")

# API endpoint
BASE_URL = f'https://graph.facebook.com/v22.0/act_{ACCOUNT_ID}/insights'

def fetch_meta_data(today):
    params = {
        'access_token': ACCESS_TOKEN,
        'fields': 'campaign_name,spend,impressions,clicks,actions,action_values',
        'time_range': f'{{"since":"{today}","until":"{today}"}}',
        'level': 'campaign',
        'limit': 100
    }
    data = []
    retries = 3
    for attempt in range(retries):
        try:
            while True:
                response = requests.get(BASE_URL, params=params)
                response.raise_for_status()
                json_response = response.json()
                if 'data' not in json_response:
                    logging.error(f"Unexpected API response: {json_response}")
                    raise Exception("Unexpected API response: Missing 'data' key")
                data.extend(json_response.get('data', []))
                logging.debug(f"Fetched {len(json_response.get('data', []))} records")
                if 'paging' in json_response and 'next' in json_response['paging']:
                    params['after'] = json_response['paging']['cursors']['after']
                else:
                    break
            if not data:
                logging.warning("No data returned from Meta API")
            return data
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:  # Rate limit
                logging.warning(f"Rate limit hit, retrying in {2 ** attempt} seconds...")
                time.sleep(2 ** attempt)
                continue
            logging.error(f"API request failed: {str(e)}")
            raise Exception(f"API request failed: {str(e)}")
        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed: {str(e)}")
            raise Exception(f"API request failed: {str(e)}")

def process_data(raw_data):
    if not raw_data:
        logging.warning("No data returned from API")
        return {
            'total_sales': 0.0,
            'total_ad_spend': 0.0,
            'overall_roas': 0.0,
            'overall_cpp': 0.0,
            'overall_ctr': 0.0,
            'overall_conversion_rate': 0.0,
            'total_impressions': 0,
            'total_clicks': 0,
            'total_conversions': 0,
            'campaign_summary': pd.DataFrame(),
            'high_roas_campaigns': pd.DataFrame(),
            'active_campaigns': pd.DataFrame()
        }
    
    df = pd.DataFrame(raw_data)
    
    # Convert data types
    df['spend'] = pd.to_numeric(df['spend'], errors='coerce').fillna(0).astype(float)
    df['impressions'] = pd.to_numeric(df['impressions'], errors='coerce').fillna(0).astype(int)
    df['clicks'] = pd.to_numeric(df['clicks'], errors='coerce').fillna(0).astype(int)
    
    # Extract conversions and sales
    def get_purchase_value(actions, key='value', action_type='purchase'):
        if isinstance(actions, list):
            for item in actions:
                if item.get('action_type') == action_type:
                    try:
                        return float(item.get(key, 0))
                    except (ValueError, TypeError):
                        logging.warning(f"Invalid {key} for {action_type}: {item.get(key)}")
                        return 0.0
        return 0.0

    df['conversions'] = df['actions'].apply(lambda x: get_purchase_value(x, 'value', 'purchase')).astype(float)
    df['sales'] = df['action_values'].apply(lambda x: get_purchase_value(x, 'value', 'purchase')).astype(float)
    
    # Calculate totals
    total_ad_spend = round(df['spend'].sum(), 2)
    total_sales = round(df['sales'].sum(), 2)
    total_impressions = df['impressions'].sum()
    total_clicks = df['clicks'].sum()
    total_conversions = df['conversions'].sum()
    
    # Calculate overall KPIs with safe division
    overall_roas = round(total_sales / total_ad_spend, 2) if total_ad_spend > 0 else 0.0
    overall_cpp = round(total_ad_spend / total_conversions, 2) if total_conversions > 0 else 0.0
    overall_ctr = round((total_clicks / total_impressions) * 100, 2) if total_impressions > 0 else 0.0
    overall_conversion_rate = round((total_conversions / total_clicks) * 100, 2) if total_clicks > 0 else 0.0
    
    # Campaign-level metrics
    campaign_summary = df.groupby('campaign_name').agg({
        'spend': 'sum',
        'sales': 'sum',
        'impressions': 'sum',
        'clicks': 'sum',
        'conversions': 'sum'
    }).reset_index()
    
    # Calculate campaign metrics with safe division
    campaign_summary['roas'] = (campaign_summary['sales'] / campaign_summary['spend']).replace([float('inf'), -float('inf')], 0).fillna(0).round(2)
    campaign_summary['cpp'] = (campaign_summary['spend'] / campaign_summary['conversions']).replace([float('inf'), -float('inf')], 0).fillna(0).round(2)
    campaign_summary['ctr'] = ((campaign_summary['clicks'] / campaign_summary['impressions']) * 100).replace([float('inf'), -float('inf')], 0).fillna(0).round(2)
    campaign_summary['conversion_rate'] = ((campaign_summary['conversions'] / campaign_summary['clicks']) * 100).replace([float('inf'), -float('inf')], 0).fillna(0).round(2)
    
    # Additional insights
    high_roas_campaigns = campaign_summary[campaign_summary['roas'] > 1]
    active_campaigns = campaign_summary[
        (campaign_summary['impressions'] > 0) | 
        (campaign_summary['clicks'] > 0) | 
        (campaign_summary['conversions'] > 0)
    ].sort_values(by='roas', ascending=False)
    
    return {
        'total_sales': total_sales,
        'total_ad_spend': total_ad_spend,
        'overall_roas': overall_roas,
        'overall_cpp': overall_cpp,
        'overall_ctr': overall_ctr,
        'overall_conversion_rate': overall_conversion_rate,
        'total_impressions': total_impressions,
        'total_clicks': total_clicks,
        'total_conversions': total_conversions,
        'campaign_summary': campaign_summary,
        'high_roas_campaigns': high_roas_campaigns,
        'active_campaigns': active_campaigns
    }

def generate_pdf_report(metrics, today, timestamp_str):
    report_name = os.path.join(REPORT_DIR, f"report_{today}.pdf")
    try:
        c = canvas.Canvas(report_name, pagesize=letter)
        width, height = letter
        
        # Define margins
        margin = 25
        table_width = width - 2 * margin
        
        # Starting Y position
        y = height - margin

        # Title with single timestamp
        c.setFont("Helvetica-Bold", 16)
        c.drawString(margin, y, f"Daily Marketing Performance Report ({timestamp_str})")
        y -= 20
        
        # Summary Metrics Table
        c.setFont("Helvetica-Bold", 14)
        c.drawString(margin, y, "Summary Metrics")
        y -= 30
        
        headers = ["Metric", "Value"]
        col_widths = [200, 200]
        x_positions = [margin, margin + col_widths[0]]
        
        # Header row
        c.setFillColorRGB(0.9, 0.9, 0.9)
        c.rect(margin, y - 5, sum(col_widths), 20, fill=True, stroke=False)
        c.setFillColorRGB(0, 0, 0)
        c.setFont("Helvetica-Bold", 12)
        for i, header in enumerate(headers):
            c.drawString(x_positions[i] + 5, y, header)
        y -= 20
        
        # Summary metrics
        summary_metrics = [
            ("Date", today),
            ("Total Sales", f"Rs {metrics['total_sales']:.2f}"),
            ("Total Ad Spend", f"Rs {metrics['total_ad_spend']:.2f}"),
            ("Overall ROAS", f"{metrics['overall_roas']:.2f}"),
            ("Overall CPP", f"Rs {metrics['overall_cpp']:.2f}"),
            ("Overall CTR", f"{metrics['overall_ctr']:.2f}%"),
            ("Overall Conversion Rate", f"{metrics['overall_conversion_rate']:.2f}%"),
            ("Total Impressions", str(int(metrics['total_impressions']))),
            ("Total Clicks", str(int(metrics['total_clicks']))),
            ("Total Conversions", str(int(metrics['total_conversions']))),
        ]
        c.setFont("Helvetica", 12)
        for metric, value in summary_metrics:
            if y < margin:
                c.showPage()
                y = height - margin
                c.setFillColorRGB(0.9, 0.9, 0.9)
                c.rect(margin, y - 5, sum(col_widths), 20, fill=True, stroke=False)
                c.setFillColorRGB(0, 0, 0)
                c.setFont("Helvetica-Bold", 12)
                for i, header in enumerate(headers):
                    c.drawString(x_positions[i] + 5, y, header)
                y -= 20
                c.setFont("Helvetica", 12)
            
            c.rect(margin, y - 5, col_widths[0], 20, stroke=True, fill=False)
            c.rect(margin + col_widths[0], y - 5, col_widths[1], 20, stroke=True, fill=False)
            c.drawString(x_positions[0] + 5, y, metric)
            c.drawString(x_positions[1] + 5, y, value)
            y -= 20
        
        # Function to wrap campaign names
        def draw_wrapped_campaign_name(canvas, name, x, y, width, font_size=10):
            canvas.setFont("Helvetica", font_size)
            words = name.split(' ')
            line = ""
            y_offset = 0
            max_lines = 2
            line_height = font_size + 1
            
            for word in words:
                test_line = line + " " + word if line else word
                if canvas.stringWidth(test_line, "Helvetica", font_size) < width - 10:
                    line = test_line
                else:
                    if y_offset == 0:
                        canvas.drawString(x + 5, y - y_offset, line)
                        line = word
                        y_offset += line_height
                    else:
                        remaining = line + " " + word
                        while canvas.stringWidth(remaining + "...", "Helvetica", font_size) > width - 10 and len(remaining) > 0:
                            remaining = remaining[:-1]
                        canvas.drawString(x + 5, y - y_offset, remaining + "..." if len(remaining) < len(line + " " + word) else remaining)
                        return
            
            if line:
                canvas.drawString(x + 5, y - y_offset, line)
        
        # Campaigns with ROAS > 1 Table
        y -= 30
        c.setFont("Helvetica-Bold", 14)
        c.drawString(margin, y, "Campaigns with ROAS > 1")
        y -= 30
        
        if not metrics['high_roas_campaigns'].empty:
            headers = ["Campaign Name", "Spend", "Sales", "ROAS", "CPP", "CTR", "CR"]
            col_widths = [200, 80, 80, 40, 80, 40, 40]
            x_positions = [margin]
            for i in range(len(col_widths) - 1):
                x_positions.append(x_positions[i] + col_widths[i])
            
            # Header row
            c.setFillColorRGB(0.9, 0.9, 0.9)
            c.rect(margin, y - 5, sum(col_widths), 20, fill=True, stroke=False)
            c.setFillColorRGB(0, 0, 0)
            c.setFont("Helvetica-Bold", 12)
            for i, header in enumerate(headers):
                c.drawString(x_positions[i] + 5, y, header)
            y -= 20
            
            for _, row in metrics['high_roas_campaigns'].iterrows():
                if y < margin + 25:
                    c.showPage()
                    y = height - margin
                    c.setFillColorRGB(0.9, 0.9, 0.9)
                    c.rect(margin, y - 5, sum(col_widths), 20, fill=True, stroke=False)
                    c.setFillColorRGB(0, 0, 0)
                    c.setFont("Helvetica-Bold", 12)
                    for i, header in enumerate(headers):
                        c.drawString(x_positions[i] + 5, y, header)
                    y -= 20
                
                row_height = 30
                
                for i in range(len(col_widths)):
                    c.rect(x_positions[i], y - row_height + 15, col_widths[i], row_height, stroke=True, fill=False)
                
                draw_wrapped_campaign_name(c, row['campaign_name'], x_positions[0], y, col_widths[0])
                
                c.setFont("Helvetica", 10)
                c.drawString(x_positions[1] + 5, y, f"Rs {row['spend']:.2f}")
                c.drawString(x_positions[2] + 5, y, f"Rs {row['sales']:.2f}")
                c.drawString(x_positions[3] + 5, y, f"{row['roas']:.2f}")
                c.drawString(x_positions[4] + 5, y, f"Rs {row['cpp']:.2f}")
                c.drawString(x_positions[5] + 5, y, f"{row['ctr']:.2f}%")
                c.drawString(x_positions[6] + 5, y, f"{row['conversion_rate']:.2f}%")
                y -= row_height
        else:
            c.setFont("Helvetica", 12)
            c.drawString(margin, y, "No campaigns with ROAS > 1")
            y -= 20
        
        # Active Campaigns Table
        y -= 30
        c.setFont("Helvetica-Bold", 14)
        c.drawString(margin, y, "Active Campaigns")
        y -= 30
        
        if not metrics['active_campaigns'].empty:
            headers = ["Campaign Name", "Spend", "Sales", "ROAS", "CPP", "CTR", "CR"]
            col_widths = [200, 80, 80, 40, 80, 40, 40]
            x_positions = [margin]
            for i in range(len(col_widths) - 1):
                x_positions.append(x_positions[i] + col_widths[i])
            
            # Header row
            c.setFillColorRGB(0.9, 0.9, 0.9)
            c.rect(margin, y - 5, sum(col_widths), 20, fill=True, stroke=False)
            c.setFillColorRGB(0, 0, 0)
            c.setFont("Helvetica-Bold", 12)
            for i, header in enumerate(headers):
                c.drawString(x_positions[i] + 5, y, header)
            y -= 20
            
            for _, row in metrics['active_campaigns'].iterrows():
                if y < margin + 25:
                    c.showPage()
                    y = height - margin
                    c.setFillColorRGB(0.9, 0.9, 0.9)
                    c.rect(margin, y - 5, sum(col_widths), 20, fill=True, stroke=False)
                    c.setFillColorRGB(0, 0, 0)
                    c.setFont("Helvetica-Bold", 12)
                    for i, header in enumerate(headers):
                        c.drawString(x_positions[i] + 5, y, header)
                    y -= 20
                
                row_height = 30
                
                for i in range(len(col_widths)):
                    c.rect(x_positions[i], y - row_height + 15, col_widths[i], row_height, stroke=True, fill=False)
                
                draw_wrapped_campaign_name(c, row['campaign_name'], x_positions[0], y, col_widths[0])
                
                c.setFont("Helvetica", 10)
                c.drawString(x_positions[1] + 5, y, f"Rs {row['spend']:.2f}")
                c.drawString(x_positions[2] + 5, y, f"Rs {row['sales']:.2f}")
                c.drawString(x_positions[3] + 5, y, f"{row['roas']:.2f}")
                c.drawString(x_positions[4] + 5, y, f"Rs {row['cpp']:.2f}")
                c.drawString(x_positions[5] + 5, y, f"{row['ctr']:.2f}%")
                c.drawString(x_positions[6] + 5, y, f"{row['conversion_rate']:.2f}%")
                y -= row_height
            
            # Summary row
            if y < margin:
                c.showPage()
                y = height - margin
            
            total_spend = round(metrics['active_campaigns']['spend'].sum(), 2)
            total_sales = round(metrics['active_campaigns']['sales'].sum(), 2)
            total_conversions = metrics['active_campaigns']['conversions'].sum()
            total_cpp = round(total_spend / total_conversions, 2) if total_conversions > 0 else 0.0
            
            c.setFillColorRGB(0.95, 0.95, 0.95)
            c.rect(margin, y - 5, sum(col_widths), 20, fill=True, stroke=False)
            c.setFillColorRGB(0, 0, 0)
            c.setFont("Helvetica-Bold", 10)
            c.drawString(x_positions[0] + 5, y, "Total")
            c.drawString(x_positions[1] + 5, y, f"Rs {total_spend:.2f}")
            c.drawString(x_positions[2] + 5, y, f"Rs {total_sales:.2f}")
            c.drawString(x_positions[4] + 5, y, f"Rs {total_cpp:.2f}")
            for i in range(len(col_widths)):
                c.rect(x_positions[i], y - 5, col_widths[i], 20, stroke=True, fill=False)
            y -= 20
        else:
            c.setFont("Helvetica", 12)
            c.drawString(margin, y, "No active campaigns")
            y -= 20
        
        c.save()
        logging.info(f"PDF report saved to: {report_name}")
        return report_name
    except IOError as e:
        logging.error(f"Failed to write PDF report: {str(e)}")
        raise Exception(f"Failed to write PDF report: {str(e)}")

def send_email(report_file, today, timestamp_str):
    retries = 3
    delay = 5
    for attempt in range(retries):
        try:
            msg = MIMEMultipart()
            msg['From'] = EMAIL_SENDER
            msg['To'] = ", ".join(EMAIL_RECIPIENTS)
            msg['Subject'] = f"Daily Marketing Report - {today} ({timestamp_str})"
            
            body = f"Attached is the daily marketing performance report for {today} (Generated at {timestamp_str})."
            msg.attach(MIMEText(body, 'plain'))
            
            with open(report_file, "rb") as f:
                part = MIMEApplication(f.read(), Name=os.path.basename(report_file))
                part['Content-Disposition'] = f'attachment; filename="{os.path.basename(report_file)}"'
                msg.attach(part)
            
            with smtplib.SMTP('smtp.gmail.com', 587, timeout=30) as server:
                server.starttls()
                server.login(EMAIL_SENDER, EMAIL_PASSWORD)
                server.sendmail(EMAIL_SENDER, EMAIL_RECIPIENTS, msg.as_string())
            
            logging.info("Email sent successfully")

            # ✅ Delete report after successful sending
            if os.path.exists(report_file):
                os.remove(report_file)
                logging.info(f"Deleted report file after sending: {report_file}")

            return
        except Exception as e:
            logging.warning(f"Email attempt {attempt + 1} failed: {str(e)}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                logging.error(f"Failed to send email after {retries} attempts: {str(e)}")
                raise Exception(f"Failed to send email: {str(e)}")


@app.route('/')
def index():
    return jsonify({
        'status': 'running',
        'service': 'meta-report-service',
        'environment': ENV
    })

@app.route('/generate-report', methods=['GET'])
def generate_report():
    try:
        today, timestamp_str = get_current_timestamp()  # GET fresh timestamp
        
        logging.info(f"Report generation started at {timestamp_str}...")

        data = fetch_meta_data(today)
        metrics = process_data(data)
        report_file = generate_pdf_report(metrics, today, timestamp_str)
        send_email(report_file, today, timestamp_str)

        logging.info("Report generation and email completed successfully.")
        return jsonify({
            'status': 'success',
            'message': 'Report generated and sent via email.',
            'report_file': report_file
        })
    except Exception as e:
        logging.error(f"Error during report generation: {str(e)}")
        traceback_str = traceback.format_exc()
        return jsonify({
            'status': 'error',
            'message': str(e),
            'trace': traceback_str
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    if ENV == 'production':
        app.run(host='0.0.0.0', port=port)
    else:
        app.run(debug=True, port=port)