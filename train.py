import re
import math
import random
from collections import Counter

############################################
# Paramètres du modèle et de l'entraînement
############################################
EMBEDDING_DIM = 64
MAX_SEQ_LEN = 10
BATCH_SIZE = 1
LEARNING_RATE = 0.1
EPOCHS = 2
MIN_COUNT = 5
N_HEADS = 4
FF_HIDDEN = 128
DROPOUT_RATE = 0.0
SEED = 1234

random.seed(SEED)

CORPUS_FILE = 'my_corpus.txt'

############################################
# Chargement du corpus
############################################
def load_corpus(filename):
    """
    Charge et tokenize le texte à partir d'un fichier.

    Args:
        filename (str): Chemin vers le fichier du corpus.

    Returns:
        list: Liste de tokens en minuscules extraits du corpus.
    """
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read().lower()
    tokens = re.findall(r'\w+', text)
    return tokens

tokens = load_corpus(CORPUS_FILE)

############################################
# Construction du vocabulaire
############################################
word_freq = Counter(tokens)
vocab = [w for w, c in word_freq.items() if c >= MIN_COUNT]
vocab = sorted(vocab)
word_to_id = {w: i for i, w in enumerate(vocab)}
id_to_word = {i: w for w, i in word_to_id.items()}
V = len(vocab)
print("Taille du vocabulaire:", V)

filtered_tokens = [w for w in tokens if w in word_to_id]

############################################
# Préparation des données
############################################
data = []
for i in range(len(filtered_tokens) - MAX_SEQ_LEN):
    input_seq = filtered_tokens[i:i + MAX_SEQ_LEN]
    target = filtered_tokens[i + MAX_SEQ_LEN]
    input_ids = [word_to_id[w] for w in input_seq]
    target_id = word_to_id[target]
    data.append((input_ids, target_id))
print("Nombre d'exemples:", len(data))

############################################
# Initialisation des paramètres du modèle
############################################
head_dim = EMBEDDING_DIM // N_HEADS
if EMBEDDING_DIM % N_HEADS != 0:
    raise ValueError("EMBEDDING_DIM doit être divisible par N_HEADS")

def random_matrix(rows, cols):
    """
    Génère une matrice aléatoire avec des valeurs initialisées entre -0.005 et 0.005.

    Args:
        rows (int): Nombre de lignes de la matrice.
        cols (int): Nombre de colonnes de la matrice.

    Returns:
        list: Matrice sous forme de liste de listes.
    """
    return [[(random.random() - 0.5) * 0.01 for _ in range(cols)] for _ in range(rows)]

word_embeddings = [[(random.random() - 0.5) * 0.01 for _ in range(EMBEDDING_DIM)] for _ in range(V)]
pos_embeddings = [[(random.random() - 0.5) * 0.01 for _ in range(EMBEDDING_DIM)] for _ in range(MAX_SEQ_LEN)]

WQ = [random_matrix(EMBEDDING_DIM, head_dim) for _ in range(N_HEADS)]
WK = [random_matrix(EMBEDDING_DIM, head_dim) for _ in range(N_HEADS)]
WV = [random_matrix(EMBEDDING_DIM, head_dim) for _ in range(N_HEADS)]
WO = random_matrix(N_HEADS * head_dim, EMBEDDING_DIM)

W1 = random_matrix(EMBEDDING_DIM, FF_HIDDEN)
b1 = [(random.random() - 0.5) * 0.01 for _ in range(FF_HIDDEN)]
W2 = random_matrix(FF_HIDDEN, EMBEDDING_DIM)
b2 = [(random.random() - 0.5) * 0.01 for _ in range(EMBEDDING_DIM)]

gamma1 = [1.0 for _ in range(EMBEDDING_DIM)]
beta1 = [0.0 for _ in range(EMBEDDING_DIM)]
gamma2 = [1.0 for _ in range(EMBEDDING_DIM)]
beta2 = [0.0 for _ in range(EMBEDDING_DIM)]

W_out = random_matrix(EMBEDDING_DIM, V)
b_out = [(random.random() - 0.5) * 0.01 for _ in range(V)]

############################################
# Fonctions utilitaires math / linear
############################################

def vector_add(u, v):
    """
    Additionne deux vecteurs élément par élément.

    Args:
        u (list): Premier vecteur.
        v (list): Deuxième vecteur.

    Returns:
        list: Vecteur résultant de l'addition.
    """
    return [a + b for a, b in zip(u, v)]

def vector_sub(u, v):
    """
    Soustrait deux vecteurs élément par élément.

    Args:
        u (list): Premier vecteur.
        v (list): Deuxième vecteur.

    Returns:
        list: Vecteur résultant de la soustraction.
    """
    return [a - b for a, b in zip(u, v)]

def vector_mul(u, v):
    """
    Multiplie deux vecteurs élément par élément.

    Args:
        u (list): Premier vecteur.
        v (list): Deuxième vecteur.

    Returns:
        list: Vecteur résultant de la multiplication.
    """
    return [a * b for a, b in zip(u, v)]

def vector_scalar_mul(u, s):
    """
    Multiplie un vecteur par un scalaire.

    Args:
        u (list): Vecteur.
        s (float): Scalaire.

    Returns:
        list: Vecteur résultant de la multiplication.
    """
    return [a * s for a in u]

def vector_dot(u, v):
    """
    Calcule le produit scalaire de deux vecteurs.

    Args:
        u (list): Premier vecteur.
        v (list): Deuxième vecteur.

    Returns:
        float: Produit scalaire.
    """
    return sum(a * b for a, b in zip(u, v))

def matvec(m, v):
    """
    Multiplie une matrice par un vecteur.

    Args:
        m (list of lists): Matrice.
        v (list): Vecteur.

    Returns:
        list: Résultat de la multiplication.
    """
    return [sum(m_i[j] * v[j] for j in range(len(v))) for m_i in m]

def softmax(logits):
    """
    Applique la fonction softmax sur une liste de logits.

    Args:
        logits (list): Liste de valeurs réelles.

    Returns:
        list: Probabilités après softmax.
    """
    max_val = max(logits)
    exps = [math.exp(x - max_val) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]

def softmax_backward(dy, probs):
    """
    Calcule le gradient de la fonction softmax.

    Args:
        dy (list): Gradient de la perte par rapport aux sorties de softmax.
        probs (list): Probabilités après softmax.

    Returns:
        list: Gradient par rapport aux logits.
    """
    dx = []
    for i in range(len(probs)):
        sum_j = sum(dy[j] * probs[j] for j in range(len(probs)))
        dx_i = probs[i] * (dy[i] - sum_j)
        dx.append(dx_i)
    return dx

def layer_norm(x, gamma, beta, eps=1e-5):
    """
    Applique la normalisation de couche sur un vecteur.

    Args:
        x (list): Vecteur d'entrée.
        gamma (list): Facteurs de mise à l'échelle.
        beta (list): Décalages.
        eps (float, optional): Petite constante pour la stabilité numérique. Defaults to 1e-5.

    Returns:
        list: Vecteur normalisé.
    """
    mean = sum(x) / len(x)
    var = sum((xi - mean) ** 2 for xi in x) / len(x)
    return [((xi - mean) / math.sqrt(var + eps)) * gamma[i] + beta[i] for i, xi in enumerate(x)]

def relu(x):
    """
    Applique la fonction ReLU sur un vecteur.

    Args:
        x (list): Vecteur d'entrée.

    Returns:
        list: Vecteur après application de ReLU.
    """
    return [max(0, xi) for xi in x]

def transpose(m):
    """
    Transpose une matrice.

    Args:
        m (list of lists): Matrice à transposer.

    Returns:
        list of lists: Matrice transposée.
    """
    rows = len(m)
    cols = len(m[0])
    return [[m[i][j] for i in range(rows)] for j in range(cols)]

def matrix_multiply(A, B):
    """
    Multiplie deux matrices A et B.

    Args:
        A (list of lists): Première matrice.
        B (list of lists): Deuxième matrice.

    Returns:
        list of lists: Résultat de la multiplication.
    """
    ra = len(A)
    ca = len(A[0])
    rb = len(B)
    cb = len(B[0])
    if ca != rb:
        raise ValueError("Nombre de colonnes de A doit être égal au nombre de lignes de B.")
    R = []
    for i in range(ra):
        row = []
        for j in range(cb):
            val = 0.0
            for k in range(ca):
                val += A[i][k] * B[k][j]
            row.append(val)
        R.append(row)
    return R

############################################
# Stockage pour forward/backward
############################################
_forward_cache = {}

############################################
# Forward Pass
############################################
def forward_pass(input_ids):
    """
    Exécute une passe avant à travers le modèle.

    Args:
        input_ids (list): Liste des identifiants de tokens (longueur = MAX_SEQ_LEN).

    Returns:
        tuple: (Sortie de la couche feed-forward, logits)
    """
    # input_ids: liste de tokens (length = MAX_SEQ_LEN)
    seq_len = len(input_ids)

    # Embeddings
    x_seq = []
    for i, wid in enumerate(input_ids):
        we = word_embeddings[wid]
        pe = pos_embeddings[i]
        x = [we[j] + pe[j] for j in range(EMBEDDING_DIM)]
        x_seq.append(x)

    # Multi-head Self-Attention
    # Q,K,V pour chaque tête
    Qs = []
    Ks = []
    Vs = []
    for h in range(N_HEADS):
        Q_h = [matvec(WQ[h], x) for x in x_seq]
        K_h = [matvec(WK[h], x) for x in x_seq]
        V_h = [matvec(WV[h], x) for x in x_seq]
        Qs.append(Q_h)
        Ks.append(K_h)
        Vs.append(V_h)

    # Scores, softmax
    attn_scores = []
    attn_weights = []
    head_outputs = []
    for h in range(N_HEADS):
        scores = []
        for i in range(seq_len):
            row_scores = []
            for j in range(seq_len):
                row_scores.append(vector_dot(Qs[h][i], Ks[h][j]) / math.sqrt(head_dim))
            scores.append(row_scores)
        w = [softmax(row) for row in scores]
        attn_weights.append(w)
        # Compute head output = w * V
        ho = []
        for i in range(seq_len):
            o = [0.0] * head_dim
            for j in range(seq_len):
                for k in range(head_dim):
                    o[k] += w[i][j] * Vs[h][j][k]
            ho.append(o)
        head_outputs.append(ho)
        attn_scores.append(scores)

    # Concat heads
    attn_out = []
    for i in range(seq_len):
        merged = []
        for h in range(N_HEADS):
            merged.extend(head_outputs[h][i])
        attn_out.append(merged)

    # WO projection
    attn_proj = []
    for i in range(seq_len):
        o = matvec(WO, attn_out[i])
        # Residual + LayerNorm
        res = [o[j] + x_seq[i][j] for j in range(EMBEDDING_DIM)]
        ln = layer_norm(res, gamma1, beta1)
        attn_proj.append(ln)

    # Feed-Forward
    ff_out = []
    ff_intermediate = []  # pour stocker avant relu
    for i in range(seq_len):
        h1 = matvec(W1, attn_proj[i])
        h1 = vector_add(h1, b1)
        h1_pre_relu = h1[:]
        h1 = relu(h1)
        ff_intermediate.append(h1_pre_relu)
        h2 = matvec(W2, h1)
        h2 = vector_add(h2, b2)
        # Residual + LayerNorm
        res2 = vector_add(h2, attn_proj[i])
        ln2 = layer_norm(res2, gamma2, beta2)
        ff_out.append(ln2)

    # Sortie
    last_vec = ff_out[-1]
    logits = matvec(transpose(W_out), last_vec)
    logits = vector_add(logits, b_out)

    # Stocker dans _forward_cache pour le backward
    _forward_cache['input_ids'] = input_ids
    _forward_cache['x_seq'] = x_seq
    _forward_cache['Q'] = Qs
    _forward_cache['K'] = Ks
    _forward_cache['V'] = Vs
    _forward_cache['attn_scores'] = attn_scores
    _forward_cache['attn_weights'] = attn_weights
    _forward_cache['head_outputs'] = head_outputs
    _forward_cache['attn_out'] = attn_out
    _forward_cache['attn_proj'] = attn_proj
    _forward_cache['ff_out'] = ff_out
    _forward_cache['ff_intermediate'] = ff_intermediate
    _forward_cache['last_vec'] = last_vec
    _forward_cache['logits'] = logits

    return ff_out, logits

def cross_entropy_loss(logits, target_id):
    """
    Calcule la perte d'entropie croisée entre les logits et la cible.

    Args:
        logits (list): Logits du modèle.
        target_id (int): Identifiant de la cible.

    Returns:
        tuple: (Perte, probabilités après softmax)
    """
    probs = softmax(logits)
    loss = -math.log(probs[target_id] + 1e-9)
    return loss, probs

def forward_for_training(input_ids, target_id):
    """
    Exécute une passe avant et calcule la perte pour l'entraînement.

    Args:
        input_ids (list): Liste des identifiants de tokens.
        target_id (int): Identifiant de la cible.

    Returns:
        float: Valeur de la perte.
    """
    ff_out, logits = forward_pass(input_ids)
    loss, probs = cross_entropy_loss(logits, target_id)
    _forward_cache['probs'] = probs
    _forward_cache['target_id'] = target_id
    return loss

############################################
# Backward Pass
############################################
def backward_for_training():
    """
    Exécute une passe arrière pour calculer les gradients et mettre à jour les paramètres du modèle.
    """
    # Récupération des valeurs du forward
    input_ids = _forward_cache['input_ids']
    x_seq = _forward_cache['x_seq']
    Qs = _forward_cache['Q']
    Ks = _forward_cache['K']
    Vs = _forward_cache['V']
    attn_scores = _forward_cache['attn_scores']
    attn_weights = _forward_cache['attn_weights']
    head_outputs = _forward_cache['head_outputs']
    attn_out = _forward_cache['attn_out']
    attn_proj = _forward_cache['attn_proj']
    ff_out = _forward_cache['ff_out']
    ff_intermediate = _forward_cache['ff_intermediate']
    last_vec = _forward_cache['last_vec']
    logits = _forward_cache['logits']
    probs = _forward_cache['probs']
    target_id = _forward_cache['target_id']

    seq_len = len(input_ids)

    # dLoss/dlogits
    dlogits = probs[:]
    dlogits[target_id] -= 1.0

    # Gradients initiaux
    dW_out = [[0.0] * V for _ in range(EMBEDDING_DIM)]
    db_out = [0.0] * V
    dlast_vec = [0.0] * EMBEDDING_DIM

    # Backprop à travers la couche de sortie
    # logits = last_vec * W_out + b_out
    for v_i in range(V):
        db_out[v_i] += dlogits[v_i]
        for d in range(EMBEDDING_DIM):
            dW_out[d][v_i] += last_vec[d] * dlogits[v_i]
            dlast_vec[d] += W_out[d][v_i] * dlogits[v_i]

    # Mettre à jour W_out, b_out plus tard, après l'accumulation des gradients
    # On continue le backprop

    # Backprop dans le FFN et layernorm final
    # ff_out[-1] = LN(res2)
    # res2 = h2 + attn_proj[-1]
    # h2 = W2 * relu(W1*attn_proj + b1) + b2

    # On doit backpropager à travers layernorm final du dernier token
    dff_out = [[0.0] * EMBEDDING_DIM for _ in range(seq_len)]
    dff_out[-1] = [d for d in dlast_vec]

    # Layernorm backward (final)
    # On a gamma2, beta2
    # LN: y = (x - mean)/sqrt(var+eps)*gamma2 + beta2
    # On a dff_out sur ln2 = y
    # Il faut backprop sur res2.
    def layernorm_backward(dy, x, gamma, beta):
        """
        Calcule le gradient de la normalisation de couche.

        Args:
            dy (list): Gradient de la perte par rapport à la sortie de la couche de normalisation.
            x (list): Entrée originale de la normalisation de couche.
            gamma (list): Facteurs de mise à l'échelle.
            beta (list): Décalages.
            eps (float, optional): Petite constante pour la stabilité numérique.

        Returns:
            tuple: (Gradient par rapport à x, gradient par rapport à gamma, gradient par rapport à beta)
        """
        L = len(x)
        mean = sum(x) / L
        var = sum((xi - mean) ** 2 for xi in x) / L
        std = math.sqrt(var + 1e-5)

        # Gradients
        dxhat = [dy[i] * gamma[i] for i in range(L)]
        dvar = sum(dxhat[i] * (x[i] - mean) * (-0.5) * (var + 1e-5) ** (-1.5) for i in range(L))
        dmean = sum(dxhat[i] * (-1.0 / std) for i in range(L)) + dvar * sum(-2.0 * (x[i] - mean) for i in range(L)) / L
        dx = [(dxhat[i] * (1.0 / std) + dvar * 2.0 * (x[i] - mean) / L + dmean / L) for i in range(L)]

        dgamma = [0.0] * L
        dbeta = [0.0] * L
        xhat = [(x[i] - mean) / std for i in range(L)]
        for i in range(L):
            dgamma[i] = dy[i] * xhat[i]
            dbeta[i] = dy[i]
        return dx, dgamma, dbeta

    # Backprop final LN
    # Pour chaque token, on backprop, mais surtout le dernier token.
    # Cependant, pour la propreté, on le fera pour tous (c'est un layer norm par token)
    dattn_proj = [[0.0] * EMBEDDING_DIM for _ in range(seq_len)]
    dgamma2 = [0.0] * EMBEDDING_DIM
    dbeta2 = [0.0] * EMBEDDING_DIM

    for i in range(seq_len):
        res2 = [h2[j] + attn_proj[i][j] for j in range(EMBEDDING_DIM)]
        dx, dg, db = layernorm_backward(dff_out[i], res2, gamma2, beta2)
        for j in range(EMBEDDING_DIM):
            dattn_proj[i][j] += dx[j]
            dgamma2[j] += dg[j]
            dbeta2[j] += db[j]

    # Maintenant backprop dans le FFN
    # res2 = h2 + attn_proj[i]
    # h2 = W2 * relu(W1*attn_proj[i] + b1) + b2
    dW2 = [[0.0] * EMBEDDING_DIM for _ in range(FF_HIDDEN)]
    db2_ = [0.0] * EMBEDDING_DIM
    dW1 = [[0.0] * FF_HIDDEN for _ in range(EMBEDDING_DIM)]
    db1_ = [0.0] * FF_HIDDEN

    dattn_proj_ff = [[0.0] * EMBEDDING_DIM for _ in range(seq_len)]

    for i in range(seq_len):
        # backward h2
        h1_pre_relu = ff_intermediate[i]
        h1 = relu(h1_pre_relu)
        # h2 = W2*h1 + b2
        dh2 = [dattn_proj[i][j] for j in range(EMBEDDING_DIM)]

        # dW2, db2
        for out_i in range(EMBEDDING_DIM):
            db2_[out_i] += dh2[out_i]
            for fh_i in range(FF_HIDDEN):
                dW2[fh_i][out_i] += h1[fh_i] * dh2[out_i]

        # dh1
        dh1 = [0.0] * FF_HIDDEN
        for fh_i in range(FF_HIDDEN):
            for out_i in range(EMBEDDING_DIM):
                dh1[fh_i] += W2[fh_i][out_i] * dh2[out_i]

        # relu backward
        for fh_i in range(FF_HIDDEN):
            if h1_pre_relu[fh_i] <= 0:
                dh1[fh_i] = 0.0

        # h1 = W1*attn_proj[i] + b1
        # dW1, db1
        ap = attn_proj[i]
        for emb_i in range(EMBEDDING_DIM):
            for fh_i in range(FF_HIDDEN):
                dW1[emb_i][fh_i] += ap[emb_i] * dh1[fh_i]
        for fh_i in range(FF_HIDDEN):
            db1_[fh_i] += dh1[fh_i]

        # dattn_proj_ff[i] = W1^T * dh1
        for emb_i in range(EMBEDDING_DIM):
            for fh_i in range(FF_HIDDEN):
                dattn_proj_ff[i][emb_i] += W1[emb_i][fh_i] * dh1[fh_i]

    # Maintenant, attn_proj[i] = LN(res)
    # res = WO*attn_out[i] + x_seq[i]
    # On a dattn_proj = dattn_proj + dattn_proj_ff
    for i in range(seq_len):
        for j in range(EMBEDDING_DIM):
            dattn_proj[i][j] += dattn_proj_ff[i][j]

    # Backprop layernorm post-attn
    dgamma1 = [0.0] * EMBEDDING_DIM
    dbeta1 = [0.0] * EMBEDDING_DIM
    dattn_out = [[0.0] * EMBEDDING_DIM for _ in range(seq_len)]
    dx_seq = [[0.0] * EMBEDDING_DIM for _ in range(seq_len)]

    for i in range(seq_len):
        res = [0.0] * EMBEDDING_DIM
        tmp = matvec(WO, attn_out[i])
        for j in range(EMBEDDING_DIM):
            res[j] = tmp[j] + x_seq[i][j]

        dx, dg, db = layernorm_backward(dattn_proj[i], res, gamma1, beta1)
        for j in range(EMBEDDING_DIM):
            dattn_out[i][j] += dx[j]
            dgamma1[j] += dg[j]
            dbeta1[j] += db[j]

        # res = WO*attn_out[i] + x_seq[i]
        # d(res) = dx
        # dx_seq[i] += dx, d(WO*attn_out[i]) += dx
        for j in range(EMBEDDING_DIM):
            dx_seq[i][j] += dx[j]

    # Backprop WO
    dWO = [[0.0] * EMBEDDING_DIM for _ in range(N_HEADS * head_dim)]
    dattn_heads_concat = [[0.0] * (N_HEADS * head_dim) for _ in range(seq_len)]
    for i in range(seq_len):
        dx = dattn_out[i]
        # attn_out[i] dimension EMBEDDING_DIM (concat de N_HEADS heads)
        # dWO = attn_out[i] * dx^T
        for in_i in range(N_HEADS * head_dim):
            for out_i in range(EMBEDDING_DIM):
                dWO[in_i][out_i] += attn_out[i][in_i] * dx[out_i]

        # dattn_out[i] -> d(attn_out[i])
        # d(attn_out[i]) = WO^T * dx
        for in_i in range(N_HEADS * head_dim):
            for out_i in range(EMBEDDING_DIM):
                dattn_heads_concat[i][in_i] += WO[in_i][out_i] * dx[out_i]

    # Séparer par tête
    dheads = []
    for h in range(N_HEADS):
        dheads.append([[0.0] * head_dim for _ in range(seq_len)])
    for i in range(seq_len):
        idx = 0
        for h in range(N_HEADS):
            for hd in range(head_dim):
                dheads[h][i][hd] = dattn_heads_concat[i][idx]
                idx += 1

    # Backprop dans l'attention
    # out_h = attn_weights * V_h
    # attn_weights = softmax(scores)
    # scores = QK^T/sqrt(head_dim)
    # Q = WQ*x_seq, etc.

    dV = [[[0.0] * head_dim for _ in range(seq_len)] for _ in range(N_HEADS)]
    dWQ = [[[0.0] * head_dim for _ in range(EMBEDDING_DIM)] for _ in range(N_HEADS)]
    dWK = [[[0.0] * head_dim for _ in range(EMBEDDING_DIM)] for _ in range(N_HEADS)]
    dWV = [[[0.0] * head_dim for _ in range(EMBEDDING_DIM)] for _ in range(N_HEADS)]

    dQ = [[[0.0] * head_dim for _ in range(seq_len)] for _ in range(N_HEADS)]
    dK = [[[0.0] * head_dim for _ in range(seq_len)] for _ in range(N_HEADS)]
    dX = [[0.0] * EMBEDDING_DIM for _ in range(seq_len)]  # pour accumuler gradient sur x_seq[i]

    for h in range(N_HEADS):
        # dheads[h] = d(out_h)
        # out_h[i] = sum_j attn_weights[h][i][j] * V_h[j]
        # d(attn_weights[h][i][j]) = sum_k dheads[h][i][k] * V_h[j][k]
        # d(V_h[j][k]) = sum_i dheads[h][i][k] * attn_weights[h][i][j]

        # Calcul des gradients par rapport à V_h
        for i in range(seq_len):
            for j in range(seq_len):
                for hd in range(head_dim):
                    dV[h][j][hd] += dheads[h][i][hd] * attn_weights[h][i][j]

        # Calcul des gradients par rapport aux scores
        for i in range(seq_len):
            # Gradient par rapport aux attn_weights
            dAttn = [0.0] * seq_len
            for j in range(seq_len):
                for hd in range(head_dim):
                    dAttn[j] += dheads[h][i][hd] * Vs[h][j][hd]

            # Gradient de la softmax
            dy = dAttn
            probs = attn_weights[h][i]
            dscores = softmax_backward(dy, probs)

            for j in range(seq_len):
                for hd in range(head_dim):
                    dQ[h][i][hd] += dscores[j] * Ks[h][j][hd]
                    dK[h][j][hd] += dscores[j] * Qs[h][i][hd]

        # Calcul des gradients par rapport aux poids WQ, WK, WV
        for i in range(seq_len):
            for hd in range(head_dim):
                for emb_i in range(EMBEDDING_DIM):
                    dWQ[h][emb_i][hd] += x_seq[i][emb_i] * dQ[h][i][hd]
                    dWK[h][emb_i][hd] += x_seq[i][emb_i] * dK[h][i][hd]
                    dWV[h][emb_i][hd] += x_seq[i][emb_i] * dV[h][i][hd]

        # Calcul des gradients par rapport à l'entrée x_seq
        for i in range(seq_len):
            for emb_i in range(EMBEDDING_DIM):
                for hd in range(head_dim):
                    dX[i][emb_i] += WQ[h][emb_i][hd] * dQ[h][i][hd]
                    dX[i][emb_i] += WK[h][emb_i][hd] * dK[h][i][hd]
                    dX[i][emb_i] += WV[h][emb_i][hd] * dV[h][i][hd]

    # Total gradient sur x_seq
    # On a dX de l'attention et dx_seq de la partie résiduelle
    # Total: dx_seq[i] += dX[i]
    for i in range(seq_len):
        for emb_i in range(EMBEDDING_DIM):
            dx_seq[i][emb_i] += dX[i][emb_i]

    # Backprop sur les embeddings
    # x_seq[i] = word_embeddings[input_ids[i]] + pos_embeddings[i]
    dword_embeddings = [[0.0] * EMBEDDING_DIM for _ in range(V)]
    dpos_embeddings = [[0.0] * EMBEDDING_DIM for _ in range(MAX_SEQ_LEN)]

    for i in range(seq_len):
        wid = input_ids[i]
        for emb_i in range(EMBEDDING_DIM):
            dword_embeddings[wid][emb_i] += dx_seq[i][emb_i]
            dpos_embeddings[i][emb_i] += dx_seq[i][emb_i]

    ###############################
    # Mise à jour des paramètres
    ###############################

    # W_out, b_out
    for v_i in range(V):
        b_out[v_i] -= LEARNING_RATE * db_out[v_i]
        for d in range(EMBEDDING_DIM):
            W_out[d][v_i] -= LEARNING_RATE * dW_out[d][v_i]

    # Layernorm gammas et betas
    for i in range(EMBEDDING_DIM):
        gamma2[i] -= LEARNING_RATE * dgamma2[i]
        beta2[i] -= LEARNING_RATE * dbeta2[i]
        gamma1[i] -= LEARNING_RATE * dgamma1[i]
        beta1[i] -= LEARNING_RATE * dbeta1[i]

    # FFN W1, b1, W2, b2
    for emb_i in range(EMBEDDING_DIM):
        for fh_i in range(FF_HIDDEN):
            W1[emb_i][fh_i] -= LEARNING_RATE * dW1[emb_i][fh_i]
    for fh_i in range(FF_HIDDEN):
        b1[fh_i] -= LEARNING_RATE * db1_[fh_i]

    for fh_i in range(FF_HIDDEN):
        for out_i in range(EMBEDDING_DIM):
            W2[fh_i][out_i] -= LEARNING_RATE * dW2[fh_i][out_i]
    for out_i in range(EMBEDDING_DIM):
        b2[out_i] -= LEARNING_RATE * db2_[out_i]

    # WO
    for in_i in range(N_HEADS * head_dim):
        for out_i in range(EMBEDDING_DIM):
            WO[in_i][out_i] -= LEARNING_RATE * dWO[in_i][out_i]

    # WQ, WK, WV
    for h in range(N_HEADS):
        for emb_i in range(EMBEDDING_DIM):
            for hd in range(head_dim):
                WQ[h][emb_i][hd] -= LEARNING_RATE * dWQ[h][emb_i][hd]
                WK[h][emb_i][hd] -= LEARNING_RATE * dWK[h][emb_i][hd]
                WV[h][emb_i][hd] -= LEARNING_RATE * dWV[h][emb_i][hd]

    # word_embeddings, pos_embeddings
    for i in range(V):
        for d in range(EMBEDDING_DIM):
            word_embeddings[i][d] -= LEARNING_RATE * dword_embeddings[i][d]

    for i in range(MAX_SEQ_LEN):
        for d in range(EMBEDDING_DIM):
            pos_embeddings[i][d] -= LEARNING_RATE * dpos_embeddings[i][d]

############################################
# Entraînement
############################################
print("Début de l'entraînement")
for epoch in range(EPOCHS):
    random.shuffle(data)
    total_loss = 0.0
    count = 0
    for (inp, tgt) in data:
        loss = forward_for_training(inp, tgt)
        total_loss += loss
        count += 1
        backward_for_training()
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss moyenne: {total_loss / (count + 1e-9):.6f}")

############################################
# Sauvegarde des embeddings
############################################
with open("attention_word_embeddings.txt", "w", encoding="utf-8") as f:
    f.write(f"{V} {EMBEDDING_DIM}\n")
    for i, w in enumerate(vocab):
        vals = " ".join(f"{x:.6f}" for x in word_embeddings[i])
        f.write(f"{w} {vals}\n")

print("Entraînement terminé et embeddings sauvegardés dans attention_word_embeddings.txt")
