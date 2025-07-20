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
# Saída: norm_points (pontos normalizados) na forma Homogêna
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

    x = [point[0] for point in points] #Obtêm os valores de x's
    y = [point[1] for point in points] #Obtêm os valores de y's
    cx = np.mean(x) #Computando o centróide de x
    cy = np.mean(y) #Computando o centróide de y
    d = 0 #Fator de escala
    # Cálculando o fator de escala s para a matriz T
    for point in points:
        d += math.sqrt((point[0]-cx)**2+(point[1]-cy)**2)
    
    num_points = len(points)
    if num_points > 0:
        d = d/num_points
    else:
        d = 0

    if d != 0:
        s = np.sqrt(2)/d
    else:
        s=1
    #Computando a matriz T de normalização do DLT

    T = np.array([[s,0,-s*cx],[0,s,-s*cy],[0,0,1]]) #Matriz de normalização antes de ser construída
    
    #Normalizando os pointos
    for point in points:
        # Cria o vetor homogêneo para o ponto atual no formato [x, y, 1]
        hom_point = np.array([point[0], point[1], 1])
        
        # Aplica a transformação T ao ponto homogêneo
        transformed_point = np.dot(T, hom_point)
        
        # Adiciona o ponto transformado à lista de resultados
        norm_points.append(transformed_point)

    return np.array(norm_points), T

# Função para montar a matriz A do sistema de equações do DLT
# Entrada: pts1, pts2 (pontos "pts1" da primeira imagem e pontos "pts2" da segunda imagem que atendem a pts2=H.pts1)
# Saída: A (matriz com as duas ou três linhas resultantes da relação pts2 x H.pts1 = 0)
def compute_A(pts1, pts2):
    """
    Função que será utilizada para montar a matriz A dado um conjunto de pontos pts1 e outro conjunto de pontos pts2.
    args:
        pts1: Matriz de pontos onde cada linha tem x,y da primeira imagem.
        pts2: Matriz de pontos onde cada linah tempontos x,y da segunda imagem.
    return:
        A: Matriz com o sistema formado para se obter a solução numérica da homografia no DLT.
    """

    A = []
    # Compute a Matriz A
    for i in range(pts1.shape[0]):
        #Itera por cada linha e pega obtem os valores de cada ponto
        x1,y1,w1 = pts1[i]#Primeira Imagem x
        x2,y2,w2 = pts2[i]#Segunda Imagem x linha
        a1 = np.array([0,0,0,-w2*x1,-w2*y1,-w2*w1,y2*x1,y2*y1,y2*w1])
        a2 = np.array([w2*x1,w2*y1,w2*w1,0,0,0,-x2*x1,-x2*y1,-x2*w1])
        a3 = np.array([-y2*x1,-y2*y1,-y2*w1,x2*x1,x2*y1,x2*w1,0,0,0])
        A.append(a1)
        A.append(a2)
        A.append(a3)
    A = np.array(A)
    return A


# Função para verificar se pontos são colineares
# Entrada: 


# Função do DLT Normalizado
# Entrada: pts1, pts2 (pontos "pts1" da primeira imagem e pontos "pts2" da segunda imagem que atendem a pts2=H.pts1)
# Saída: H (matriz de homografia estimada)
def compute_normalized_dlt(pts1, pts2):
    """
    Função Responsável por normalizar os pontos pts1 e pts2.
    args:
        pts1: Matriz de pontos onde cada linha tem pares de  pontos  x,y da primeira imagem.
        pts2: Matriz de pontos onde cada linha tem pares de  pontos x,y da segunda imagem.
    return:
        H: Matriz de homografia estimada após os pontos pts1 e pts2 serem normalizados.
    """

    # Normaliza pontos
    pts1,T1 = normalize_points(pts1)
    pts2,T2 = normalize_points(pts2)
    # Constrói o sistema de equações empilhando a matrix A de cada par de pontos correspondentes normalizados
    A = compute_A(pts1,pts2)
    # Calcula o SVD da matriz A_empilhada e estima a homografia H_normalizada 
    U,S,Vt = np.linalg.svd(A)
    # Remodela a última coluna de V como uma matriz de homografia
    H_norm = Vt[-1,:].reshape(3,3)# ~H
    # Denormaliza H_normalizada e obtém H
    T2_inv = np.linalg.inv(T2)
    H = np.dot(T2_inv,H_norm)#Computa T2_inv * ~H 
    H = np.dot(H,T1)#Computa (T2_inv * ~H) * T1

    #Normalização
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
    
    # Alteração feita pelo gemini
    # Explicação: Os parâmetros agora são definidos dentro da função, como em seu esqueleto.
    # `best_H` e `max_inliers` são inicializados para guardar o melhor modelo encontrado.
    s = 4                # Tamanho da amostra, como definido por você.
    p = 0.99             # Probabilidade de sucesso, como definido por você.
    sample_count = 0     # Contador de iterações.
    n_total = len(pts1)  # Número total de pontos.
    
    best_H = None
    max_inliers = 0
    best_inlier_indices = []

    # Processo Iterativo
    while sample_count < N:
        # Sorteia aleatoriamente "s" amostras do conjunto de pares de pontos pts1 e pts2 
        # Alteração feita pelo gemini: A amostragem foi corrigida. O método `np.append` estava
        # incorreto. O correto é sortear os índices e usá-los para pegar os pontos
        # correspondentes de pts1 e pts2.
        indices = random.sample(range(n_total), s)
        sample_pts1 = pts1[indices]
        sample_pts2 = pts2[indices]

        # Usa as amostras para estimar uma homografia usando o DTL Normalizado
        try:
            H_curr = compute_normalized_dlt(sample_pts1, sample_pts2)
        except np.linalg.LinAlgError:
            sample_count += 1
            continue # Pula iteração se os pontos forem colineares

        # Testa essa homografia com os demais pares de pontos usando o dis_threshold
        # Alteração feita pelo gemini: A lógica para encontrar o conjunto de consenso
        # estava incorreta. `np.dot` não calcula a distância. O código abaixo implementa
        # o cálculo do erro geométrico (distância euclidiana) corretamente.
        current_inlier_indices = []
        for i in range(n_total):
            pt1_hom = np.array([pts1[i, 0], pts1[i, 1], 1])
            p2_proj_hom = np.dot(H_curr, pt1_hom)
            if p2_proj_hom[2] == 0: continue
            
            p2_proj = p2_proj_hom[:2] / p2_proj_hom[2]#Normaliza para as coordenadas estejam na forma (x,y,1)
            error = np.linalg.norm(pts2[i] - p2_proj)
            
            if error < dis_threshold:
                current_inlier_indices.append(i)
        
        # Se o número de inliers é o maior obtido até o momento, guarda esse modelo.
        # Alteração feita pelo gemini: A variável `n_inliers` não estava definida. Corrigido
        # para usar `max_inliers` e `len(current_inlier_indices)`.
        if len(current_inlier_indices) > max_inliers:
            max_inliers = len(current_inlier_indices)
            best_H = H_curr
            best_inlier_indices = current_inlier_indices
            
            # Atualiza também o número N de iterações necessárias (parte adaptativa)
            # Alteração feita pelo gemini: A fórmula foi corrigida para usar o `max_inliers`
            # correto e o operador de potência do Python (`**` em vez de `^`).
            e = 1 - (max_inliers / n_total)
            if e < 1.0: # Evita log de 0
                numerator = math.log(1 - p)
                denominator = math.log(1 - (1 - e)**s)
                if denominator != 0:
                    N_adaptive = numerator / denominator
                    # O N da iteração não deve exceder o N máximo fornecido pelo usuário
                    if N_adaptive < N:
                        N = N_adaptive

        # Critério de parada opcional com base em Ninl
        if Ninl is not None and max_inliers >= Ninl:
            break
            
        sample_count += 1

    # Terminado o processo iterativo, estima a homografia final H usando todos os inliers.
    # Alteração feita pelo gemini: Esta etapa, chamada de "refinement", estava faltando.
    # Pegamos o melhor conjunto de inliers e recalculamos a homografia para obter um resultado final mais preciso.
    final_H = None
    pts1_in, pts2_in = None, None
    if best_H is not None and max_inliers > s:
        pts1_in = pts1[best_inlier_indices]
        pts2_in = pts2[best_inlier_indices]
        final_H = compute_normalized_dlt(pts1_in, pts2_in)

    return final_H, pts1_in, pts2_in


########################################################################################################################
# Exemplo de Teste da função de homografia usando o SIFT


MIN_MATCH_COUNT = 10
img1 = cv.imread('box.jpg', 0)   # queryImage
img2 = cv.imread('photo01a.jpg', 0)        # trainImage

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

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1, 1, 2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1, 1, 2)
    
    #################################################
    M = # AQUI ENTRA A SUA FUNÇÃO DE HOMOGRAFIA!!!!
    #################################################

    img4 = cv.warpPerspective(img1, M, (img2.shape[1], img2.shape[0])) 

else:
    print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   flags = 2)
img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

fig, axs = plt.subplots(2, 2, figsize=(30, 15))
fig.add_subplot(2, 2, 1)
plt.imshow(img3, 'gray')
fig.add_subplot(2, 2, 2)
plt.title('Primeira imagem')
plt.imshow(img1, 'gray')
fig.add_subplot(2, 2, 3)
plt.title('Segunda imagem')
plt.imshow(img2, 'gray')
fig.add_subplot(2, 2, 4)
plt.title('Primeira imagem após transformação')
plt.imshow(img4, 'gray')
plt.show()

########################################################################################################################
