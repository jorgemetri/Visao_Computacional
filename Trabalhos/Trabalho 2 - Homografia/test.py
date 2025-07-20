# TRABALHO 2 DE VISÃO COMPUTACIONAL
# Nome: Jorge Metri Miranda e Rodrigo Bergamin

# Importa as bibliotecas necessárias
# Acrescente qualquer outra que quiser
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2 as cv
np.random.seed(0)#Define a semente aleatória que será utilizada para sortear os valores em S
import random

########################################################################################################################
# Função para normalizar pontos
# Entrada: points (pontos da imagem a serem normalizados)
# Saída: norm_points (pontos normalizados)
#        T (matriz de normalização)
def normalize_points(points):
    """
    Função que irá dado pontos de entrada retornar uma matriz T normalizada.
    args:
        points: Matriz de pontos onde cada linha tem um vetor x,y.
    return:
        norm_points: Vetor com pontos normalizados.
        T: Matriz de normalização.(3x3)
    """
    norm_points = []
    x = [point[0] for point in points]#Define os valores de x's
    y = [point[1] for point in points]#Define os valores de y's
    cx = np.mean(x)#Computa a média dos valores de x's
    cy = np.mean(y)#Computa a média dos valores de y's
    d = 0
    for point in points:
        d += math.sqrt((point[0]-cx)**2+(point[1]-cy)**2)#Computa da distancia euclidiana
    num_points = len(points)
    if num_points > 0:
        d = d/num_points
    else:
        d = 0
    if d != 0:
        s = np.sqrt(2)/d#Computa o fator s
    else:
        s=1
    T = np.array([[s,0,-s*cx],[0,s,-s*cy],[0,0,1]])#Monta a matriz T
    for point in points:
        hom_point = np.array([point[0], point[1], 1])#Transforma em coordenadas homogênas
        transformed_point = np.dot(T, hom_point)#Normaliza os pontos
        norm_points.append(transformed_point)
    return np.array(norm_points), T

# Função para montar a matriz A do sistema de equações do DLT
# Entrada: pts1, pts2 (pontos "pts1" da primeira imagem e pontos "pts2" da segunda imagem que atendem a pts2=H.pts1)
# Saída: A (matriz com as duas ou três linhas resultantes da relação pts2 x H.pts1 = 0)
def compute_A(pts1, pts2):
    """
    Função que será utilizada para montar a matriz A dado um conjunto de pontos pts1 e outro conjunto de pontos pts2.
    args:
        pts1: Matriz de pontos HOMOGÊNEOS (N,3).
        pts2: Matriz de pontos HOMOGÊNEOS (N,3).
    return:
        A: Matriz com o sistema formado para se obter a solução numérica da homografia no DLT.
    """
    # Conforme solicitado, esta função não foi alterada para manter as 3 equações.
    A = []
    for i in range(pts1.shape[0]):
        x1,y1,w1 = pts1[i]
        x2,y2,w2 = pts2[i]
        #Cria a matriz A_i
        a1 = np.array([0,0,0,-w2*x1,-w2*y1,-w2*w1,y2*x1,y2*y1,y2*w1])
        a2 = np.array([w2*x1,w2*y1,w2*w1,0,0,0,-x2*x1,-x2*y1,-x2*w1])
        a3 = np.array([-y2*x1,-y2*y1,-y2*w1,x2*x1,x2*y1,x2*w1,0,0,0])
        A.append(a1)
        A.append(a2)
        A.append(a3)
    return np.array(A)

# Função do DLT Normalizado
# Entrada: pts1, pts2 (pontos "pts1" da primeira imagem e pontos "pts2" da segunda imagem que atendem a pts2=H.pts1)
# Saída: H (matriz de homografia estimada)
def compute_normalized_dlt(pts1, pts2):
    """
    Função Responsável por normalizar os pontos pts1 e pts2.
    args:
        pts1: Matriz de pontos (N,2).
        pts2: Matriz de pontos (N,2).
    return:
        H: Matriz de homografia estimada.
    """
    if pts1.shape[0] < 4:
        raise ValueError("São necessários pelo menos 4 pontos para calcular a homografia.")
    norm_pts1, T1 = normalize_points(pts1)#Normaliza os pontos de pts1
    norm_pts2, T2 = normalize_points(pts2)#Normaliza os pontos de pts2
    A = compute_A(norm_pts1, norm_pts2)#Computa a matriz A
    U,S,Vt = np.linalg.svd(A)
    H_norm = Vt[-1,:].reshape(3,3)
    T2_inv = np.linalg.inv(T2)
    H = np.dot(T2_inv,H_norm)
    H = np.dot(H,T1)
    #Normaliza
    if H[2,2] != 0:
        H = H/H[2,2]
    return H

# Função do RANSAC
# Entradas:
# pts1: pontos da primeira imagem
# pts2: pontos da segunda imagem 
# dis_threshold: limiar de distância a ser usado no RANSAC
# N: número máximo de iterações (pode ser definido dentro da função e deve ser atualizado 
#    dinamicamente de acordo com o número de inliers/outliers)
# Ninl: limiar de inliers desejado (pode ser ignorado ou não - fica como decisão de vocês)
# Saídas:
# H: homografia estimada
# pts1_in, pts2_in: conjunto de inliers dos pontos da primeira e segunda imagens
def RANSAC(pts1, pts2, dis_threshold=5.99, N=500, Ninl=None):
    """
    Essa função é a função responsável por remover outliers e escolher a homografia a partir de um processo
    de iteração.
    args:
        pts1: Matriz de pontos 2d
        pts2: Maztriz de pontos 2d
        dis_threhold: Distância utilizada para verificar se um ponto é inlier ou não.
        N: Número de iterações inicias
        Ninl:Parâmetro não obrigatório.
    return:
        H: Homografia estimada pelo processo iterativo.
        pts1_in: Conjunto de inliers da primeira imagem.
        pts2_in: Conjunto de inliers da segunda imagem.
    """
    # Define outros parâmetros como número de amostras do modelo, probabilidades da equação de N, etc
    s = 4
    p = 0.99
    sample_count = 0
    n_total = len(pts1)
    
    best_H = None
    max_inliers = 0
    best_inlier_indices = []

    # Processo Iterativo
    while sample_count < N:
        # Enquanto não atende a critério de parada

        # Sorteia aleatoriamente "s" amostras do conjunto de pares de pontos pts1 e pts2 
        indices = random.sample(range(n_total), s)
        sample_pts1 = pts1[indices]
        sample_pts2 = pts2[indices]

        # Usa as amostras para estimar uma homografia usando o DTL Normalizado
        try:
            H_curr = compute_normalized_dlt(sample_pts1, sample_pts2)
        except (np.linalg.LinAlgError, ValueError):
            sample_count += 1
            continue

        current_inlier_indices = []
        # Testa essa homografia com os demais pares de pontos usando o dis_threshold e contabiliza
        for i in range(n_total):
            pt1_hom = np.array([pts1[i, 0], pts1[i, 1], 1])
            p2_proj_hom = np.dot(H_curr, pt1_hom)
            if p2_proj_hom[2] == 0: continue
            
            p2_proj = p2_proj_hom[:2] / p2_proj_hom[2]
            error = np.linalg.norm(pts2[i] - p2_proj)
            
            if error < dis_threshold:
                current_inlier_indices.append(i)
        
         # o número de supostos inliers obtidos com o modelo estimado
        if len(current_inlier_indices) > max_inliers:
            max_inliers = len(current_inlier_indices)
            best_H = H_curr
            best_inlier_indices = current_inlier_indices
            
            e = 1 - (max_inliers / n_total)
            if e < 1.0:
                numerator = math.log(1 - p)
                denominator = math.log(1 - (1 - e)**s)
                if denominator != 0:
                    N_adaptive = int(numerator / denominator)
                    if N_adaptive < N:
                        N = N_adaptive


            
        sample_count += 1
        # Terminado o processo iterativo
    final_H = None
    pts1_in, pts2_in = None, None
    # Estima a homografia final H usando todos os inliers selecionados.
    if best_H is not None and max_inliers >= s: # Usar >= s em vez de > s
        pts1_in = pts1[best_inlier_indices]
        pts2_in = pts2[best_inlier_indices]
        final_H = compute_normalized_dlt(pts1_in, pts2_in)

    return final_H, pts1_in, pts2_in

########################################################################################################################
# Exemplo de Teste da função de homografia usando o SIFT

MIN_MATCH_COUNT = 10


img1 = cv.imread('box.jpg', 0)
img2 = cv.imread('photo01a.jpg', 0)

# Inicialização do SIFT
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
# FLANN
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

img4 = np.zeros_like(img2)
M = None
M_cv = None # Inicializa a matriz do OpenCV também

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1, 1, 2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1, 1, 2)
    
    src_pts_reshaped = src_pts.reshape(-1, 2)
    dst_pts_reshaped = dst_pts.reshape(-1, 2)
    #################################################################################
    
    M, pts1_in, pts2_in = RANSAC(src_pts_reshaped, dst_pts_reshaped, dis_threshold=7.5)# AQUI ENTRA A SUA FUNÇÃO DE HOMOGRAFIA!!!!
    #####################################################################################
    # Chamada da função de homografia do OpenCV para comparação
    M_cv, mask_cv = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    
    if M is not None:
        img4 = cv.warpPerspective(img1, M, (img2.shape[1], img2.shape[0])) 
    else:
        print("RANSAC falhou em encontrar uma homografia válida.")

else:
    print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
    matchesMask = None

print("\n--- COMPARAÇÃO DE HOMOGRAFIAS ---")
if M is not None:
    print("\nSua Homografia (RANSAC implementado):\n", M)
else:
    print("\nSua Homografia (RANSAC implementado): FALHOU")
    
if M_cv is not None:
    print("\nHomografia do OpenCV:\n", M_cv)
else:
    print("\nHomografia do OpenCV: FALHOU")
print("----------------------------------\n")


#Exibição do Resultado-------------------------------------------------

draw_params = dict(matchColor = (0,255,0), # desenha as correspondências em verde
                   singlePointColor = None,
                   matchesMask = None, # desenha todas as "good" matches
                   flags = 2)
img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

fig, axs = plt.subplots(2, 2, figsize=(20, 12))
fig.suptitle('Comparação de Homografia', fontsize=20)
fig.add_subplot(2, 2, 1)
plt.imshow(img3, 'gray')
plt.title('Correspondências SIFT')
fig.add_subplot(2, 2, 2)
plt.title('Primeira imagem (template)')
plt.imshow(img1, 'gray')
fig.add_subplot(2, 2, 3)
plt.title('Segunda imagem (cena)')
plt.imshow(img2, 'gray')
fig.add_subplot(2, 2, 4)
plt.title('Primeira imagem transformada pela SUA homografia')
plt.imshow(img4, 'gray')
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajusta o layout para o supertítulo
plt.show()

########################################################################################################################```