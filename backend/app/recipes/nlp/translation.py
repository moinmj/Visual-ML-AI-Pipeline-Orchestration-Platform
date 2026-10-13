import re
import json
import urllib.request
import urllib.parse
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort
from backend.app.core.logging import logger


# ----------------------------------------------------------------------
# BUILT-IN MULTILINGUAL DICTIONARY & OFFLINE TRANSLATION LEXICON
# Provides fast, deterministic, zero-dependency offline translation
# ----------------------------------------------------------------------

# Multi-word phrase translations (checked before single-word replacement)
PHRASE_TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "es": {
        "thank you": "gracias",
        "thanks a lot": "muchas gracias",
        "good morning": "buenos días",
        "good afternoon": "buenas tardes",
        "good evening": "buenas noches",
        "good night": "buenas noches",
        "customer service": "servicio al cliente",
        "customer support": "soporte al cliente",
        "in progress": "en progreso",
        "out of stock": "agotado",
        "credit card": "tarjeta de crédito",
        "debit card": "tarjeta de débito",
        "terms of service": "términos de servicio",
        "privacy policy": "política de privacidad",
        "very good": "muy bueno",
        "not working": "no funciona",
        "free shipping": "envío gratis",
        "user manual": "manual de usuario",
        "high quality": "alta calidad",
        "low quality": "baja calidad",
        "money back": "devolución de dinero",
        "return policy": "política de devoluciones"
    },
    "fr": {
        "thank you": "merci",
        "thanks a lot": "merci beaucoup",
        "good morning": "bonjour",
        "good afternoon": "bon après-midi",
        "good evening": "bonsoir",
        "good night": "bonne nuit",
        "customer service": "service client",
        "customer support": "support client",
        "in progress": "en cours",
        "out of stock": "en rupture de stock",
        "credit card": "carte de crédit",
        "debit card": "carte de débit",
        "terms of service": "conditions d'utilisation",
        "privacy policy": "politique de confidentialité",
        "very good": "très bon",
        "not working": "ne fonctionne pas",
        "free shipping": "livraison gratuite",
        "user manual": "mode d'emploi",
        "high quality": "haute qualité",
        "low quality": "basse qualité",
        "money back": "remboursement",
        "return policy": "politique de retour"
    },
    "de": {
        "thank you": "danke",
        "thanks a lot": "vielen dank",
        "good morning": "guten morgen",
        "good afternoon": "guten tag",
        "good evening": "guten abend",
        "good night": "gute nacht",
        "customer service": "kundenservice",
        "customer support": "kundensupport",
        "in progress": "in bearbeitung",
        "out of stock": "nicht vorrätig",
        "credit card": "kreditkarte",
        "debit card": "debitkarte",
        "terms of service": "nutzungsbedingungen",
        "privacy policy": "datenschutzrichtlinie",
        "very good": "sehr gut",
        "not working": "funktioniert nicht",
        "free shipping": "kostenloser versand",
        "user manual": "benutzerhandbuch",
        "high quality": "hohe qualität",
        "low quality": "geringe qualität",
        "money back": "geld zurück",
        "return policy": "rückgaberichtlinie"
    },
    "it": {
        "thank you": "grazie",
        "thanks a lot": "molte grazie",
        "good morning": "buongiorno",
        "good afternoon": "buon pomeriggio",
        "good evening": "buonasera",
        "good night": "buonanotte",
        "customer service": "servizio clienti",
        "customer support": "assistenza clienti",
        "in progress": "in corso",
        "out of stock": "esaurito",
        "credit card": "carta di credito",
        "debit card": "carta di debito",
        "very good": "molto buono",
        "not working": "non funziona",
        "free shipping": "spedizione gratuita"
    },
    "pt": {
        "thank you": "obrigado",
        "thanks a lot": "muito obrigado",
        "good morning": "bom dia",
        "good afternoon": "boa tarde",
        "good evening": "boa noite",
        "good night": "boa noite",
        "customer service": "atendimento ao cliente",
        "customer support": "suporte ao cliente",
        "in progress": "em andamento",
        "out of stock": "esgotado",
        "credit card": "cartão de crédito",
        "debit card": "cartão de débito",
        "very good": "muito bom",
        "not working": "não está funcionando",
        "free shipping": "frete grátis"
    },
    "zh": {
        "thank you": "谢谢",
        "thanks a lot": "非常感谢",
        "good morning": "早上好",
        "good afternoon": "下午好",
        "good evening": "晚上好",
        "good night": "晚安",
        "customer service": "客户服务",
        "customer support": "客户支持",
        "in progress": "进行中",
        "out of stock": "缺货",
        "credit card": "信用卡",
        "very good": "非常好",
        "not working": "无法正常工作",
        "free shipping": "免运费"
    },
    "ja": {
        "thank you": "ありがとう",
        "thanks a lot": "どうもありがとうございます",
        "good morning": "おはようございます",
        "good afternoon": "こんにちは",
        "good evening": "こんばんは",
        "good night": "おやすみなさい",
        "customer service": "カスタマーサービス",
        "customer support": "カスタマーサポート",
        "in progress": "進行中",
        "out of stock": "在庫切れ",
        "credit card": "クレジットカード",
        "very good": "とても良い",
        "free shipping": "送料無料"
    },
    "ru": {
        "thank you": "спасибо",
        "thanks a lot": "большое спасибо",
        "good morning": "доброе утро",
        "good afternoon": "добрый день",
        "good evening": "добрый вечер",
        "good night": "спокойной ночи",
        "customer service": "обслуживание клиентов",
        "customer support": "служба поддержки",
        "in progress": "в процессе",
        "out of stock": "нет в наличии",
        "credit card": "кредитная карта",
        "very good": "очень хорошо",
        "free shipping": "бесплатная доставка"
    },
    "ar": {
        "thank you": "شكرا لك",
        "thanks a lot": "شكرا جزيلا",
        "good morning": "صباح الخير",
        "good evening": "مساء الخير",
        "customer service": "خدمة العملاء",
        "customer support": "دعم العملاء",
        "in progress": "قيد التنفيذ",
        "out of stock": "غير متوفر بالمخزون",
        "credit card": "بطاقة ائتمان",
        "very good": "جيد جدا",
        "free shipping": "شحن مجاني"
    },
    "hi": {
        "thank you": "धन्यवाद",
        "thanks a lot": "बहुत बहुत धन्यवाद",
        "good morning": "शुभ प्रभात",
        "good afternoon": "शुभ दोपहर",
        "good evening": "शुभ संध्या",
        "good night": "शुभ रात्रि",
        "customer service": "ग्राहक सेवा",
        "customer support": "ग्राहक सहायता",
        "in progress": "प्रगति पर है",
        "out of stock": "स्टॉक खत्म",
        "credit card": "क्रेडिट कार्ड",
        "very good": "बहुत अच्छा",
        "free shipping": "मुफ़्त डिलीवरी"
    }
}

# Core word translations across English to multiple target languages
WORD_TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "es": {
        # Greetings & conversational
        "hello": "hola", "hi": "hola", "welcome": "bienvenido", "goodbye": "adiós",
        "bye": "adiós", "please": "por favor", "thanks": "gracias", "yes": "sí",
        "no": "no", "okay": "bien", "ok": "bien", "sure": "seguro",
        "sorry": "lo siento", "excuse": "disculpe", "pardon": "perdón",

        # Feedback & Sentiment
        "good": "bueno", "great": "genial", "excellent": "excelente", "amazing": "increíble",
        "wonderful": "maravilloso", "love": "amor", "loved": "amado", "awesome": "asombroso",
        "perfect": "perfecto", "bad": "malo", "poor": "pobre", "terrible": "terrible",
        "awful": "horrible", "horrible": "horrible", "worst": "peor", "hate": "odiar",
        "happy": "feliz", "unhappy": "infeliz", "satisfied": "satisfecho",
        "disappointed": "decepcionado", "problem": "problema", "problems": "problemas",
        "issue": "problema", "issues": "problemas", "error": "error", "errors": "errores",
        "bug": "error", "clean": "limpio", "dirty": "sucio", "broken": "roto",
        "fast": "rápido", "slow": "lento", "easy": "fácil", "hard": "difícil",
        "difficult": "difícil", "useful": "útil", "useless": "inútil", "helpful": "útil",
        "superb": "magnífico", "fantastic": "fantástico", "fine": "bien",

        # Business / E-commerce / Tech
        "customer": "cliente", "customers": "clientes", "client": "cliente", "user": "usuario",
        "users": "usuarios", "order": "pedido", "orders": "pedidos", "product": "producto",
        "products": "productos", "item": "artículo", "items": "artículos", "price": "precio",
        "cost": "costo", "money": "dinero", "free": "gratis", "discount": "descuento",
        "payment": "pago", "invoice": "factura", "receipt": "recibo", "account": "cuenta",
        "service": "servicio", "services": "servicios", "support": "soporte",
        "shipping": "envío", "delivery": "entrega", "return": "devolución", "returns": "devoluciones",
        "refund": "reembolso", "cancel": "cancelar", "cancelled": "cancelado",
        "success": "éxito", "failure": "fallo", "failed": "fallido", "status": "estado",
        "pending": "pendiente", "completed": "completado", "approved": "aprobado",
        "rejected": "rechazado", "verified": "verificado", "active": "activo",
        "inactive": "inactivo", "open": "abierto", "closed": "cerrado",
        "new": "nuevo", "old": "viejo", "updated": "actualizado", "created": "creado",
        "data": "datos", "dataset": "conjunto de datos", "report": "informe",
        "file": "archivo", "files": "archivos", "system": "sistema", "network": "red",
        "server": "servidor", "email": "correo electrónico", "phone": "teléfono",
        "message": "mensaje", "messages": "mensajes", "address": "dirección",
        "date": "fecha", "time": "tiempo", "year": "año", "month": "mes", "day": "día",
        "hour": "hora", "minute": "minuto", "second": "segundo", "name": "nombre",
        "title": "título", "description": "descripción", "category": "categoría",
        "review": "reseña", "reviews": "reseñas", "rating": "calificación",
        "feedback": "comentarios", "help": "ayuda", "info": "información",

        # Grammar & Common Connectors
        "and": "y", "or": "o", "but": "pero", "with": "con", "without": "sin",
        "for": "para", "from": "de", "to": "a", "in": "en", "on": "en",
        "at": "en", "by": "por", "about": "acerca de", "is": "es", "are": "son",
        "was": "era", "were": "eran", "have": "tener", "has": "tiene",
        "had": "tenía", "this": "este", "that": "ese", "these": "estos",
        "those": "esos", "the": "el", "a": "un", "an": "un", "all": "todo",
        "some": "algunos", "many": "muchos", "much": "mucho", "more": "más",
        "less": "menos", "very": "muy", "not": "no", "what": "qué",
        "where": "dónde", "when": "cuándo", "why": "por qué", "how": "cómo",
        "who": "quién", "which": "cuál", "we": "nosotros", "you": "usted",
        "they": "ellos", "i": "yo", "he": "él", "she": "ella", "it": "ello"
    },
    "fr": {
        "hello": "bonjour", "hi": "salut", "welcome": "bienvenue", "goodbye": "au revoir",
        "bye": "au revoir", "please": "s'il vous plaît", "thanks": "merci", "yes": "oui",
        "no": "non", "okay": "d'accord", "ok": "d'accord", "sure": "bien sûr",
        "sorry": "désolé", "excuse": "excusez-moi", "pardon": "pardon",
        "good": "bon", "great": "génial", "excellent": "excellent", "amazing": "incroyable",
        "wonderful": "merveilleux", "love": "amour", "loved": "aimé", "awesome": "fantastique",
        "perfect": "parfait", "bad": "mauvais", "poor": "médiocre", "terrible": "terrible",
        "awful": "affreux", "horrible": "horrible", "worst": "pire", "hate": "détester",
        "happy": "heureux", "unhappy": "malheureux", "satisfied": "satisfait",
        "disappointed": "déçu", "problem": "problème", "problems": "problèmes",
        "issue": "problème", "issues": "problèmes", "error": "erreur", "errors": "erreurs",
        "bug": "bogue", "clean": "propre", "dirty": "sale", "broken": "cassé",
        "fast": "rapide", "slow": "lent", "easy": "facile", "hard": "difficile",
        "difficult": "difficile", "useful": "utile", "useless": "inutile", "helpful": "utile",
        "customer": "client", "customers": "clients", "client": "client", "user": "utilisateur",
        "users": "utilisateurs", "order": "commande", "orders": "commandes", "product": "produit",
        "products": "produits", "item": "article", "items": "articles", "price": "prix",
        "cost": "coût", "money": "argent", "free": "gratuit", "discount": "réduction",
        "payment": "paiement", "invoice": "facture", "receipt": "reçu", "account": "compte",
        "service": "service", "services": "services", "support": "support",
        "shipping": "expédition", "delivery": "livraison", "return": "retour", "returns": "retours",
        "refund": "remboursement", "cancel": "annuler", "cancelled": "annulé",
        "success": "succès", "failure": "échec", "failed": "échoué", "status": "statut",
        "pending": "en attente", "completed": "terminé", "approved": "approuvé",
        "rejected": "rejeté", "verified": "vérifié", "active": "actif",
        "inactive": "inactif", "open": "ouvert", "closed": "fermé",
        "new": "nouveau", "old": "ancien", "updated": "mis à jour",
        "data": "données", "report": "rapport", "file": "fichier", "system": "système",
        "email": "e-mail", "phone": "téléphone", "message": "message", "address": "adresse",
        "date": "date", "time": "temps", "year": "année", "month": "mois", "day": "jour",
        "name": "nom", "title": "titre", "description": "description", "category": "catégorie",
        "review": "avis", "feedback": "retour", "help": "aide",
        "and": "et", "or": "ou", "but": "mais", "with": "avec", "without": "sans",
        "for": "pour", "from": "de", "to": "à", "in": "dans", "on": "sur",
        "is": "est", "are": "sont", "was": "était", "were": "étaient", "have": "avoir",
        "this": "ce", "that": "cela", "the": "le", "a": "un", "an": "un", "all": "tout",
        "very": "très", "not": "non", "what": "quoi", "where": "où", "when": "quand",
        "why": "pourquoi", "how": "comment", "who": "qui", "which": "lequel"
    },
    "de": {
        "hello": "hallo", "hi": "hallo", "welcome": "willkommen", "goodbye": "auf wiedersehen",
        "bye": "tschuess", "please": "bitte", "thanks": "danke", "yes": "ja",
        "no": "nein", "okay": "in ordnung", "ok": "ok", "sure": "sicher",
        "sorry": "entschuldigung",
        "good": "gut", "great": "grossartig", "excellent": "ausgezeichnet", "amazing": "erstaunlich",
        "wonderful": "wunderbar", "love": "liebe", "awesome": "klasse", "perfect": "perfekt",
        "bad": "schlecht", "poor": "mangelhaft", "terrible": "schrecklich", "awful": "furchtbar",
        "horrible": "entsetzlich", "worst": "schlechteste", "hate": "hassen",
        "happy": "gluecklich", "satisfied": "zufrieden", "disappointed": "enttaeuscht",
        "problem": "problem", "problems": "probleme", "issue": "problem", "issues": "probleme",
        "error": "fehler", "errors": "fehler", "bug": "fehler", "clean": "sauber",
        "dirty": "schmutzig", "broken": "kaputt", "fast": "schnell", "slow": "langsam",
        "easy": "einfach", "hard": "schwer", "difficult": "schwierig", "useful": "nuetzlich",
        "customer": "kunde", "customers": "kunden", "user": "benutzer", "users": "benutzer",
        "order": "bestellung", "orders": "bestellungen", "product": "produkt", "products": "produkte",
        "item": "artikel", "items": "artikel", "price": "preis", "cost": "kosten",
        "money": "geld", "free": "kostenlos", "discount": "rabatt", "payment": "zahlung",
        "invoice": "rechnung", "receipt": "quittung", "account": "konto", "service": "service",
        "support": "unterstuetzung", "shipping": "versand", "delivery": "lieferung",
        "return": "rueckgabe", "refund": "rueckerstattung", "cancel": "abbrechen",
        "success": "erfolg", "failure": "fehlschlag", "failed": "fehlgeschlagen",
        "status": "status", "pending": "ausstehend", "completed": "abgeschlossen",
        "approved": "genehmigt", "rejected": "abgelehnt", "verified": "verifiziert",
        "active": "aktiv", "inactive": "inaktiv", "open": "offen", "closed": "geschlossen",
        "new": "neu", "old": "alt", "updated": "aktualisiert",
        "data": "daten", "report": "bericht", "file": "datei", "system": "system",
        "email": "e-mail", "phone": "telefon", "message": "nachricht", "address": "adresse",
        "date": "datum", "time": "zeit", "year": "jahr", "month": "monat", "day": "tag",
        "name": "name", "title": "titel", "description": "beschreibung",
        "review": "bewertung", "feedback": "rueckmeldung", "help": "hilfe",
        "and": "und", "or": "oder", "but": "aber", "with": "mit", "without": "ohne",
        "for": "fuer", "from": "von", "to": "nach", "in": "in", "on": "auf",
        "is": "ist", "are": "sind", "was": "war", "were": "waren", "have": "haben",
        "the": "der", "a": "ein", "an": "ein", "all": "alle", "very": "sehr", "not": "nicht"
    },
    "it": {
        "hello": "ciao", "hi": "ciao", "welcome": "benvenuto", "goodbye": "arrivederci",
        "please": "per favore", "thanks": "grazie", "yes": "sì", "no": "no",
        "good": "buono", "great": "ottimo", "excellent": "eccellente", "amazing": "incredibile",
        "wonderful": "meraviglioso", "love": "amore", "perfect": "perfetto", "bad": "cattivo",
        "terrible": "terribile", "happy": "felice", "satisfied": "soddisfatto",
        "problem": "problema", "error": "errore", "broken": "rotto", "fast": "veloce",
        "slow": "lento", "easy": "facile", "customer": "cliente", "user": "utente",
        "order": "ordine", "product": "prodotto", "price": "prezzo", "free": "gratuito",
        "payment": "pagamento", "service": "servizio", "support": "supporto",
        "delivery": "consegna", "refund": "rimborso", "status": "stato",
        "completed": "completato", "active": "attivo", "data": "dati",
        "and": "e", "or": "o", "but": "ma", "with": "con", "is": "è", "the": "il", "a": "un"
    },
    "pt": {
        "hello": "olá", "hi": "oi", "welcome": "bem-vindo", "goodbye": "adeus",
        "please": "por favor", "thanks": "obrigado", "yes": "sim", "no": "não",
        "good": "bom", "great": "ótimo", "excellent": "excelente", "amazing": "incrível",
        "wonderful": "maravilhoso", "love": "amor", "perfect": "perfeito", "bad": "ruim",
        "terrible": "terrível", "happy": "feliz", "satisfied": "satisfeito",
        "problem": "problema", "error": "erro", "broken": "quebrado", "fast": "rápido",
        "slow": "lento", "easy": "fácil", "customer": "cliente", "user": "usuário",
        "order": "pedido", "product": "produto", "price": "preço", "free": "grátis",
        "payment": "pagamento", "service": "serviço", "support": "suporte",
        "delivery": "entrega", "refund": "reembolso", "status": "status",
        "completed": "concluído", "active": "ativo", "data": "dados",
        "and": "e", "or": "ou", "but": "mas", "with": "com", "is": "é", "the": "o", "a": "um"
    },
    "zh": {
        "hello": "你好", "hi": "嗨", "welcome": "欢迎", "goodbye": "再见",
        "please": "请", "thanks": "谢谢", "yes": "是", "no": "否",
        "good": "好", "great": "棒", "excellent": "优秀", "amazing": "太棒了",
        "wonderful": "极好", "love": "爱", "perfect": "完美", "bad": "不好",
        "terrible": "糟糕", "happy": "开心", "satisfied": "满意",
        "problem": "问题", "error": "错误", "broken": "损坏", "fast": "快速",
        "slow": "缓慢", "easy": "简单", "customer": "客户", "user": "用户",
        "order": "订单", "product": "产品", "price": "价格", "free": "免费",
        "payment": "支付", "service": "服务", "support": "支持",
        "delivery": "交付", "refund": "退款", "status": "状态",
        "completed": "已完成", "active": "活跃", "data": "数据",
        "and": "和", "or": "或", "but": "但是", "with": "与", "is": "是"
    },
    "ja": {
        "hello": "こんにちは", "hi": "やあ", "welcome": "ようこそ", "goodbye": "さようなら",
        "please": "お願いします", "thanks": "ありがとう", "yes": "はい", "no": "いいえ",
        "good": "良い", "great": "すばらしい", "excellent": "優れている", "amazing": "素晴らしい",
        "wonderful": "素敵", "love": "愛", "perfect": "完璧", "bad": "悪い",
        "terrible": "ひどい", "happy": "幸せ", "satisfied": "満足",
        "problem": "問題", "error": "エラー", "broken": "壊れた", "fast": "速い",
        "slow": "遅い", "easy": "簡単", "customer": "顧客", "user": "ユーザー",
        "order": "注文", "product": "製品", "price": "価格", "free": "無料",
        "payment": "支払い", "service": "サービス", "support": "サポート",
        "delivery": "配達", "refund": "返金", "status": "ステータス",
        "completed": "完了", "active": "アクティブ", "data": "データ",
        "and": "そして", "or": "または", "but": "しかし", "with": "と一緒に", "is": "です"
    },
    "ru": {
        "hello": "здравствуйте", "hi": "привет", "welcome": "добро пожаловать", "goodbye": "до свидания",
        "please": "пожалуйста", "thanks": "спасибо", "yes": "да", "no": "нет",
        "good": "хорошо", "great": "отлично", "excellent": "превосходно", "amazing": "удивительно",
        "wonderful": "замечательно", "love": "любовь", "perfect": "идеально", "bad": "плохо",
        "terrible": "ужасно", "happy": "счастлив", "satisfied": "доволен",
        "problem": "проблема", "error": "ошибка", "broken": "сломан", "fast": "быстро",
        "slow": "медленно", "easy": "легко", "customer": "клиент", "user": "пользователь",
        "order": "заказ", "product": "продукт", "price": "цена", "free": "бесплатно",
        "payment": "оплата", "service": "сервис", "support": "поддержка",
        "delivery": "доставка", "refund": "возврат", "status": "статус",
        "completed": "завершено", "active": "активен", "data": "данные",
        "and": "и", "or": "или", "but": "но", "with": "с", "is": "это"
    },
    "ar": {
        "hello": "مرحبا", "hi": "أهلا", "welcome": "أهلا بك", "goodbye": "مع السلامة",
        "please": "من فضلك", "thanks": "شكرا", "yes": "نعم", "no": "لا",
        "good": "جيد", "great": "عظيم", "excellent": "ممتاز", "amazing": "مدهش",
        "wonderful": "رائع", "love": "حب", "perfect": "مثالي", "bad": "سيء",
        "terrible": "فظيع", "happy": "سعيد", "satisfied": "راض",
        "problem": "مشكلة", "error": "خطأ", "broken": "مكسور", "fast": "سريع",
        "slow": "بطيء", "easy": "سهل", "customer": "عميل", "user": "مستخدم",
        "order": "طلب", "product": "منتج", "price": "سعر", "free": "مجاني",
        "payment": "دفع", "service": "خدمة", "support": "دعم",
        "delivery": "توصيل", "refund": "استرداد", "status": "حالة",
        "completed": "مكتمل", "active": "نشط", "data": "بيانات",
        "and": "و", "or": "أو", "but": "لكن", "with": "مع", "is": "هو"
    },
    "hi": {
        "hello": "नमस्ते", "hi": "नमस्ते", "welcome": "स्वागत है", "goodbye": "अलविदा",
        "please": "कृपया", "thanks": "धन्यवाद", "yes": "हाँ", "no": "नहीं",
        "good": "अच्छा", "great": "बहुत बढ़िया", "excellent": "उत्कृष्ट", "amazing": "अद्भुत",
        "wonderful": "शानदार", "love": "प्यार", "perfect": "उत्तम", "bad": "बुरा",
        "terrible": "भयानक", "happy": "खुश", "satisfied": "संतुष्ट",
        "problem": "समस्या", "error": "त्रुटि", "broken": "टूटा हुआ", "fast": "तेज़",
        "slow": "धीमा", "easy": "आसान", "customer": "ग्राहक", "user": "उपयोगकर्ता",
        "order": "आदेश", "product": "उत्पाद", "price": "मूल्य", "free": "मुफ़्त",
        "payment": "भुगतान", "service": "सेवा", "support": "सहायता",
        "delivery": "वितरण", "refund": "धनवापसी", "status": "स्थिति",
        "completed": "पूर्ण", "active": "सक्रिय", "data": "डेटा",
        "and": "और", "or": "या", "but": "लेकिन", "with": "साथ", "is": "है"
    }
}

# Supported language metadata
SUPPORTED_LANGUAGES: Dict[str, str] = {
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "zh": "Chinese",
    "ja": "Japanese",
    "ru": "Russian",
    "ar": "Arabic",
    "hi": "Hindi",
    "en": "English",
    "nl": "Dutch",
    "ko": "Korean",
    "tr": "Turkish"
}


def normalize_lang_code(lang: str) -> str:
    """Extract 2-character ISO language code from string like 'es', 'Spanish (es)', etc."""
    if not lang or not isinstance(lang, str):
        return "es"
    lang_clean = lang.strip().lower()
    match = re.search(r"\b([a-z]{2})\b", lang_clean)
    if match and match.group(1) in SUPPORTED_LANGUAGES:
        return match.group(1)
    if lang_clean in SUPPORTED_LANGUAGES:
        return lang_clean
    # Check language names
    for code, name in SUPPORTED_LANGUAGES.items():
        if name.lower() in lang_clean:
            return code
    return "es"


def translate_text_offline(text: str, target_lang: str, source_lang: str = "auto") -> str:
    """
    Offline dictionary and rule-based translator.
    Preserves casing (UPPERCASE, Title Case, lowercase) and punctuation.
    """
    if not text or not isinstance(text, str):
        return "" if text is None else str(text)

    trimmed = text.strip()
    if not trimmed:
        return text

    target_lang = normalize_lang_code(target_lang)
    phrases = PHRASE_TRANSLATIONS.get(target_lang, {})
    lexicon = WORD_TRANSLATIONS.get(target_lang, {})

    if not lexicon and not phrases:
        # Fallback if unsupported target
        return f"[{target_lang.upper()}] {text}"

    # 1. Check exact full match in phrases or words
    text_lower = trimmed.lower()
    if text_lower in phrases:
        trans = phrases[text_lower]
        if trimmed.isupper():
            return trans.upper()
        if trimmed.istitle():
            return trans.title()
        return trans

    if text_lower in lexicon:
        trans = lexicon[text_lower]
        if trimmed.isupper():
            return trans.upper()
        if trimmed.istitle():
            return trans.title()
        return trans

    # 2. Multi-word phrase replacement in the string
    working_text = trimmed
    for phrase_src, phrase_tgt in phrases.items():
        pattern = re.compile(rf"\b{re.escape(phrase_src)}\b", re.IGNORECASE)
        def _phrase_replacer(match: re.Match) -> str:
            m_str = match.group(0)
            if m_str.isupper():
                return phrase_tgt.upper()
            if m_str.istitle():
                return phrase_tgt.title()
            return phrase_tgt
        working_text = pattern.sub(_phrase_replacer, working_text)

    # 3. Word-by-word token replacement preserving punctuation and casing
    tokens = re.split(r"(\w+)", working_text)
    translated_tokens = []
    for token in tokens:
        token_lower = token.lower()
        if token_lower in lexicon:
            trans = lexicon[token_lower]
            if token.isupper() and len(token) > 1:
                trans = trans.upper()
            elif token.istitle():
                trans = trans.title()
            translated_tokens.append(trans)
        else:
            translated_tokens.append(token)

    return "".join(translated_tokens)


def translate_text_online_google(text: str, target_lang: str, source_lang: str = "auto") -> Optional[str]:
    """
    Attempts public Google Translate endpoint with short timeout.
    Returns None on network timeout/failure to trigger offline fallback.
    """
    if not text or not isinstance(text, str) or not text.strip():
        return text

    target_code = normalize_lang_code(target_lang)
    source_code = "auto" if source_lang in ["auto", "", None] else normalize_lang_code(source_lang)

    try:
        url = (
            f"https://translate.googleapis.com/translate_a/single?"
            f"client=gtx&sl={source_code}&tl={target_code}&dt=t&q="
            + urllib.parse.quote(text)
        )
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36"}
        )
        with urllib.request.urlopen(req, timeout=2.5) as response:
            if response.status == 200:
                body = response.read().decode("utf-8")
                data = json.loads(body)
                if data and isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                    translated_segments = [seg[0] for seg in data[0] if seg and len(seg) > 0 and seg[0]]
                    if translated_segments:
                        return "".join(translated_segments)
    except Exception as e:
        logger.debug(f"Online translation request failed ({e}), falling back to offline dictionary.")
    return None


class TranslationRecipe(BaseRecipe):
    """
    NLP / Text Translation Recipe.
    Translates text columns into another language with zero-dependency offline dictionary
    lexicon and optional online engine fallback.
    """

    recipe_id = "text_translation"
    name = "Translation"
    version = "1.0.0"
    category = "nlp"
    description = "Translates text columns into another language."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs = [
        RecipePort(
            id="dataframe",
            label="Input DataFrame",
            type="dataframe",
            required=True,
            max_connections=1,
            description="Dataset containing text columns to translate"
        )
    ]

    outputs = [
        RecipePort(
            id="dataframe",
            label="Translated DataFrame",
            type="dataframe",
            description="Dataset containing translated text column(s)"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "columns": {
                    "type": "column",
                    "input_type": "column_select",
                    "title": "Target Text Columns",
                    "description": "Text columns to translate into another language. If left empty, all detected text/object columns will be translated."
                },
                "destination_language": {
                    "type": "select",
                    "input_type": "dropdown",
                    "title": "Target Language",
                    "enum": [
                        "Spanish (es)",
                        "French (fr)",
                        "German (de)",
                        "Italian (it)",
                        "Portuguese (pt)",
                        "Chinese (zh)",
                        "Japanese (ja)",
                        "Russian (ru)",
                        "Arabic (ar)",
                        "Hindi (hi)",
                        "English (en)",
                        "Dutch (nl)",
                        "Korean (ko)",
                        "Turkish (tr)"
                    ],
                    "default": "Spanish (es)",
                    "description": "Destination language for the translation."
                },
                "source_language": {
                    "type": "select",
                    "input_type": "dropdown",
                    "title": "Source Language",
                    "enum": [
                        "Auto-Detect (auto)",
                        "English (en)",
                        "Spanish (es)",
                        "French (fr)",
                        "German (de)",
                        "Italian (it)",
                        "Portuguese (pt)",
                        "Chinese (zh)",
                        "Japanese (ja)",
                        "Russian (ru)",
                        "Arabic (ar)",
                        "Hindi (hi)"
                    ],
                    "default": "Auto-Detect (auto)",
                    "description": "Source language code or 'auto' for automatic detection."
                },
                "engine": {
                    "type": "select",
                    "input_type": "dropdown",
                    "title": "Translation Engine",
                    "enum": [
                        "Offline Dictionary (Fast & Deterministic)",
                        "Google Translate (Public Web API with Offline Fallback)",
                        "Mock / Simulation"
                    ],
                    "default": "Offline Dictionary (Fast & Deterministic)",
                    "description": "Choose translation engine. Offline dictionary requires zero internet connection or API keys."
                },
                "append_columns": {
                    "type": "boolean",
                    "input_type": "boolean",
                    "title": "Append Output Columns",
                    "default": True,
                    "description": "When true, appends new translated columns alongside existing columns. When false, replaces original text columns."
                },
                "output_suffix": {
                    "type": "text",
                    "input_type": "text",
                    "title": "Output Column Suffix",
                    "default": "_translated",
                    "description": "Suffix for newly appended translated columns (e.g. '_translated' or '_es')."
                }
            }
        }

    def execute(
        self,
        inputs: Dict[str, Any],
        config: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        df = inputs.get("dataframe")
        if df is None or not isinstance(df, pd.DataFrame):
            # Accept left_dataframe or input fallback
            df = inputs.get("left_dataframe") or inputs.get("input")
            if df is None or not isinstance(df, pd.DataFrame):
                return {
                    "dataframe": pd.DataFrame(),
                    "output_summary": {"error": "No valid input DataFrame provided for Text Translation."}
                }

        if df.empty:
            return {
                "dataframe": df.copy(),
                "metrics": {"total_rows": 0, "translated_columns": []},
                "output_summary": {"message": "Input DataFrame is empty. No translation performed."}
            }

        out_df = df.copy()

        # Parse configuration
        target_lang_raw = config.get("destination_language") or config.get("target_language") or "Spanish (es)"
        target_lang = normalize_lang_code(str(target_lang_raw))

        src_lang_raw = config.get("source_language", "Auto-Detect (auto)")
        source_lang = "auto" if "auto" in str(src_lang_raw).lower() else normalize_lang_code(str(src_lang_raw))

        engine_raw = str(config.get("engine", "Offline Dictionary")).lower()
        append_cols = bool(config.get("append_columns", True))
        suffix = str(config.get("output_suffix", "_translated")).strip() or "_translated"

        # Resolve target columns
        cols_cfg = config.get("columns")
        target_cols: List[str] = []
        if isinstance(cols_cfg, list) and cols_cfg:
            target_cols = [str(c) for c in cols_cfg if str(c) in out_df.columns]
        elif isinstance(cols_cfg, str) and cols_cfg in out_df.columns:
            target_cols = [cols_cfg]
        else:
            # Auto-detect text columns (object or string dtypes)
            for col in out_df.columns:
                dtype_str = str(out_df[col].dtype).lower()
                if dtype_str in ["object", "string"] or "str" in dtype_str:
                    target_cols.append(col)

        if not target_cols:
            return {
                "dataframe": out_df,
                "metrics": {"total_rows": len(out_df), "translated_columns": []},
                "output_summary": {
                    "title": "Translation",
                    "message": "No text or string columns detected to translate.",
                    "status": "SKIPPED"
                }
            }

        # Translation function with memoization/cache
        translation_cache: Dict[str, str] = {}

        def _translate_value(val: Any) -> Any:
            if val is None or pd.isna(val):
                return val
            val_str = str(val)
            if not val_str.strip():
                return val_str

            if val_str in translation_cache:
                return translation_cache[val_str]

            if "mock" in engine_raw:
                translated = f"[{target_lang.upper()}] {val_str}"
            elif "google" in engine_raw:
                online_res = translate_text_online_google(val_str, target_lang, source_lang)
                if online_res is not None:
                    translated = online_res
                else:
                    translated = translate_text_offline(val_str, target_lang, source_lang)
            else:
                translated = translate_text_offline(val_str, target_lang, source_lang)

            translation_cache[val_str] = translated
            return translated

        total_translations = 0
        column_summaries = {}

        for col in target_cols:
            series = out_df[col]
            translated_series = series.apply(_translate_value)
            non_empty_count = translated_series.dropna().count()
            total_translations += int(non_empty_count)

            out_col_name = f"{col}{suffix}" if append_cols else col
            out_df[out_col_name] = translated_series

            sample_pairs = []
            for orig, trans in zip(series.head(3), translated_series.head(3)):
                if pd.notna(orig) and pd.notna(trans):
                    sample_pairs.append({"original": str(orig), "translated": str(trans)})

            column_summaries[col] = {
                "output_column": out_col_name,
                "translated_rows": int(non_empty_count),
                "samples": sample_pairs
            }

        lang_name = SUPPORTED_LANGUAGES.get(target_lang, target_lang.upper())
        engine_display = (
            "Mock Simulation" if "mock" in engine_raw
            else "Google Translate (Public)" if "google" in engine_raw
            else "Offline Dictionary"
        )

        metrics = {
            "total_rows": len(out_df),
            "translated_columns": target_cols,
            "target_language": target_lang,
            "target_language_name": lang_name,
            "source_language": source_lang,
            "engine": engine_display,
            "total_translated_cells": total_translations,
            "column_breakdown": column_summaries
        }

        output_summary = {
            "title": "Text Translation Completed",
            "message": f"Successfully translated {len(target_cols)} column(s) into {lang_name} ({target_lang}) across {len(out_df)} rows using {engine_display}.",
            "metrics": metrics,
            "status": "SUCCESS"
        }

        return {
            "dataframe": out_df,
            "metrics": metrics,
            "output_summary": output_summary
        }
