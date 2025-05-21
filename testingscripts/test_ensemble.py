import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.predict2 import predict_email_hybrid  # ✅ new function

# Example emails to test
emails = [
    """
    Corrupti maiores modi earum esse nisi non molestiae eveniet. à¤†à¤ªà¤•à¥‡ à¤–à¤¾à¤¤à¥‡ à¤¸à¥‡ à¤•à¥à¤› à¤…à¤¨à¤§à¤¿à¤•à¥ƒà¤¤ à¤²à¥‡à¤¨-à¤¦à¥‡à¤¨ à¤¹à¥à¤ à¤¹à¥ˆà¤‚à¥¤ à¤¯à¤¦à¤¿ à¤†à¤ªà¤¨à¥‡ à¤‡à¤¸à¥‡ à¤¸à¥à¤µà¥€à¤•à¥ƒà¤¤ à¤¨à¤¹à¥€à¤‚ à¤•à¤¿à¤¯à¤¾ à¤¹à¥ˆ, à¤¤à¥‹ à¤¤à¤¤à¥à¤•à¤¾à¤² à¤¹à¤®à¤¾à¤°à¥‡ à¤¸à¥à¤°à¤•à¥à¤·à¤¿à¤¤ à¤ªà¥‹à¤°à¥à¤Ÿà¤² à¤ªà¤° à¤²à¥‰à¤— à¤‡à¤¨ à¤•à¤°à¤•à¥‡ à¤ªà¥à¤·à¥à¤Ÿà¤¿ à¤•à¤°à¥‡à¤‚à¥¤ à¤¸à¤‚à¤¦à¤°à¥à¤­: 6258
    """,
    """
    Objet : Confirmation de votre réservation

Bonjour Monsieur Dupont,

Nous vous confirmons la réservation de votre séjour à l'hôtel Le Soleil du 12 mai au 15 mai 2025.

Votre numéro de réservation est le #HTL-982374. Vous pouvez présenter ce numéro à la réception lors de votre arrivée. Le check-in est à partir de 14h00.

Pour toute question ou demande particulière, n'hésitez pas à nous contacter par retour de mail ou par téléphone au +33 1 23 45 67 89.

Nous vous remercions pour votre confiance et vous souhaitons un excellent séjour.

Cordialement,  
L’équipe de l’Hôtel Le Soleil
    """,
    """
    bug uri processing turns uuencoded strings into http uris which then causes fps changed what removed added otherbugsdependingo nthis receiving mail because assignee bug watching assignee
    """,
    """
    è¿™æ˜¯çŠ¶æ€è§£å†³ç›®å‰è¿™é‡Œæ¥æºå­¦ç”Ÿå¥³äººå¼€å§‹. æ‚¨å¥½ï¼Œç³»ç»Ÿæ£€æµ‹åˆ°æ‚¨è´¦æˆ·å­˜åœ¨é£Žé™©è¡Œä¸ºï¼Œè¯·æ‚¨ç«‹å³ç™»å½•å¹¶å®Œæˆå®‰å…¨éªŒè¯ï¼Œä»¥ä¿æŠ¤æ‚¨çš„èµ„é‡‘å®‰å…¨ã€‚ è´¦æˆ·ç¼–å·: 3127
    """
]

# Final hybrid testing
for i, email in enumerate(emails, 1):
    print("\n" + "=" * 70)
    print(f"[Email {i}] Testing email:\n{email.strip()}\n")
    
    label, votes = predict_email_hybrid(email)
    
    print(f"✅ Final Hybrid Prediction: {label}")
    print(f"🧠 Individual Model Votes (DL + ML): {votes}")
    print(f"{'⚠️  Potential conflict' if votes.count(0) != 0 and votes.count(1) != 0 else '✅ Unanimous or majority consensus'}")
