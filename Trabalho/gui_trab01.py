import sys
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QGridLayout, QLabel, QWidget, QLineEdit, QHBoxLayout, QVBoxLayout, QPushButton,QGroupBox
from PyQt5.QtGui import QDoubleValidator
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy import array
from transformacoes import *

###### Crie suas funções de translação, rotação, criação de referenciais, plotagem de setas e qualquer outra função que precisar

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        #definindo as variaveis
        self.set_variables()
        #Ajustando a tela    
        self.setWindowTitle("Grid Layout")
        self.setGeometry(100, 100,1280 , 720)
        self.setup_ui()
        #Atualiza a tela com o estado inicial
        self.update_canvas()


    def set_variables(self):
        self.objeto_original = ReturnObject1()
        self.objeto = self.objeto_original
        #A câmera original (referencial do mundo)
        self.world_frame = np.hstack((Base(),np.array([[0,0,0,1]]).T))
        #Posição inicial da câmera movida para trás para enxergar o objeto
        #self.cam_original = move(0, 0, 50)
        self.cam_original = move(20,-5,6)@Rz(pi/2)@Rx(-pi/2)
        self.cam = self.cam_original

        self.px_base = 1280
        self.px_altura = 720
        self.dist_foc = 50 
        self.stheta = 0 
        self.ox = self.px_base/2
        self.oy = self.px_altura/2
        self.ccd = [36,24]
        self.projection_matrix =  np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
        self.sx = self.px_base/self.ccd[0]
        self.sy = self.px_altura/self.ccd[1]

        print(f"""
            ======================================================
            Função: set_variables (RESET)
            ======================================================
            Parâmetros de Geometria:
            - Objeto Original:     (Shape: {self.objeto_original.shape})
            - Objeto Atual:        (Shape: {self.objeto.shape})
            - Câmera Original:     \n{self.cam_original}
            - Câmera Atual:        \n{self.cam}

            Parâmetros Intrínsecos da Câmera:
            - Resolução (w, h):    ({self.px_base}, {self.px_altura}) [px]
            - Ponto Principal (ox, oy): ({self.ox}, {self.oy}) [px]
            - Sensor CCD (x, y):   {self.ccd} [mm]
            - Distância Focal:     {self.dist_foc}
            - Skew (s_theta):      {self.stheta}
            - Skew x,y (sx,sy): {self.sx:.2f},{self.sy:.2f} [px]/[mm]

            Matrizes:
            - Matriz de Projeção:  \n{self.projection_matrix}
            ======================================================
        """)
        
        
    def setup_ui(self):
        # Criar o layout de grade
        grid_layout = QGridLayout()

        # Criar os widgets
        line_edit_widget1 = self.create_world_widget("Ref mundo")
        line_edit_widget2  = self.create_cam_widget("Ref camera")
        line_edit_widget3  = self.create_intrinsic_widget("params instr")

        self.canvas = self.create_matplotlib_canvas()

        # Adicionar os widgets ao layout de grade
        grid_layout.addWidget(line_edit_widget1, 0, 0)
        grid_layout.addWidget(line_edit_widget2, 0, 1)
        grid_layout.addWidget(line_edit_widget3, 0, 2)
        grid_layout.addWidget(self.canvas, 1, 0, 1, 3)

        # Criar um widget para agrupar o botão de reset
        reset_widget = QWidget()
        reset_layout = QHBoxLayout()
        reset_widget.setLayout(reset_layout)

        # Criar o botão de reset vermelho
        reset_button = QPushButton("Reset")
        reset_button.setFixedSize(50, 30)
        style_sheet = """
            QPushButton {
                color : white ;
                background: rgba(255, 127, 130,128);
                font: inherit;
                border-radius: 5px;
                line-height: 1;
            }
        """
        reset_button.setStyleSheet(style_sheet)
        reset_button.clicked.connect(self.reset_canvas)

        # Adicionar o botão de reset ao layout
        reset_layout.addWidget(reset_button)

        # Adicionar o widget de reset ao layout de grade
        grid_layout.addWidget(reset_widget, 2, 0, 1, 3)

        # Criar um widget central e definir o layout de grade como seu layout
        central_widget = QWidget()
        central_widget.setLayout(grid_layout)
        
        # Definir o widget central na janela principal
        self.setCentralWidget(central_widget)

    def create_intrinsic_widget(self, title):
        # Criar um widget para agrupar os QLineEdit
        line_edit_widget = QGroupBox(title)
        line_edit_layout = QVBoxLayout()
        line_edit_widget.setLayout(line_edit_layout)

        # Criar um layout de grade para dividir os QLineEdit em 3 colunas
        grid_layout = QGridLayout()

        line_edits = []
        labels = ['n_pixels_base(px):', 'n_pixels_altura(px):', 'ccd_x(mm):', 'ccd_y(mm):', 'dist_focal:', 'sθ:']  # Texto a ser exibido antes de cada QLineEdit

        # Adicionar widgets QLineEdit com caixa de texto ao layout de grade
        for i in range(1, 7):
            line_edit = QLineEdit()
            label = QLabel(labels[i-1])
            validator = QDoubleValidator()
            line_edit.setValidator(validator)
            grid_layout.addWidget(label, (i-1)//2, 2*((i-1)%2))
            grid_layout.addWidget(line_edit, (i-1)//2, 2*((i-1)%2) + 1)
            line_edits.append(line_edit)

        # Criar o botão de atualização
        update_button = QPushButton("Atualizar")
        
        # Conectar a função de atualização aos sinais de clique do botão
        update_button.clicked.connect(lambda: self.update_params_intrinsc(line_edits))

        # Adicionar os widgets ao layout do widget line_edit_widget
        line_edit_layout.addLayout(grid_layout)
        line_edit_layout.addWidget(update_button)

        return line_edit_widget
    
    def create_world_widget(self, title):
        # Criar um widget para agrupar os QLineEdit
        line_edit_widget = QGroupBox(title)
        line_edit_layout = QVBoxLayout()
        line_edit_widget.setLayout(line_edit_layout)

        # Criar um layout de grade para dividir os QLineEdit em 3 colunas
        grid_layout = QGridLayout()

        line_edits = []
        labels = ['X(move):', 'X(angle):', 'Y(move):', 'Y(angle):', 'Z(move):', 'Z(angle):']

        # Adicionar widgets QLineEdit com caixa de texto ao layout de grade
        for i in range(1, 7):
            line_edit = QLineEdit()
            label = QLabel(labels[i-1])
            validator = QDoubleValidator()
            line_edit.setValidator(validator)
            grid_layout.addWidget(label, (i-1)//2, 2*((i-1)%2))
            grid_layout.addWidget(line_edit, (i-1)//2, 2*((i-1)%2) + 1)
            line_edits.append(line_edit)

        # Criar o botão de atualização
        update_button = QPushButton("Atualizar")

        # Conectar a função de atualização aos sinais de clique do botão
        update_button.clicked.connect(lambda: self.update_world(line_edits))

        # Adicionar os widgets ao layout do widget line_edit_widget
        line_edit_layout.addLayout(grid_layout)
        line_edit_layout.addWidget(update_button)

        return line_edit_widget

    def create_cam_widget(self, title):
        # Criar um widget para agrupar os QLineEdit
        line_edit_widget = QGroupBox(title)
        line_edit_layout = QVBoxLayout()
        line_edit_widget.setLayout(line_edit_layout)

        # Criar um layout de grade para dividir os QLineEdit em 3 colunas
        grid_layout = QGridLayout()

        line_edits = []
        labels = ['X(move):', 'X(angle):', 'Y(move):', 'Y(angle):', 'Z(move):', 'Z(angle):']

        # Adicionar widgets QLineEdit com caixa de texto ao layout de grade
        for i in range(1, 7):
            line_edit = QLineEdit()
            label = QLabel(labels[i-1])
            validator = QDoubleValidator()
            line_edit.setValidator(validator)
            grid_layout.addWidget(label, (i-1)//2, 2*((i-1)%2))
            grid_layout.addWidget(line_edit, (i-1)//2, 2*((i-1)%2) + 1)
            line_edits.append(line_edit)

        # Criar o botão de atualização
        update_button = QPushButton("Atualizar")

        # Conectar a função de atualização aos sinais de clique do botão
        update_button.clicked.connect(lambda: self.update_cam(line_edits))

        # Adicionar os widgets ao layout do widget line_edit_widget
        line_edit_layout.addLayout(grid_layout)
        line_edit_layout.addWidget(update_button)

        return line_edit_widget

    def create_matplotlib_canvas(self):
        # Criar um widget para exibir os gráficos do Matplotlib
        canvas_widget = QWidget()
        canvas_layout = QHBoxLayout()
        canvas_widget.setLayout(canvas_layout)

        # Criar um objeto FigureCanvas para exibir o gráfico 2D
        self.fig1, self.ax1 = plt.subplots()
        self.canvas1 = FigureCanvas(self.fig1)
        canvas_layout.addWidget(self.canvas1)

        # Criar um objeto FigureCanvas para exibir o gráfico 3D
        self.fig2 = plt.figure()
        self.ax2 = self.fig2.add_subplot(111, projection='3d')
        self.canvas2 = FigureCanvas(self.fig2)
        canvas_layout.addWidget(self.canvas2)

        return canvas_widget

    ##### Você deverá criar as suas funções aqui
    
    def update_params_intrinsc(self, line_edits):
        """
        Método Responsável para alterar os parâmetros intrísicos fornecido pelo usuario
        """
        try:
            px_base_text = line_edits[0].text()
            if px_base_text: self.px_base = float(px_base_text)         
            
            px_altura_text = line_edits[1].text()
            if px_altura_text: self.px_altura = float(px_altura_text)

            ccd_x_text = line_edits[2].text()
            if ccd_x_text: self.ccd[0] = float(ccd_x_text)

            ccd_y_text = line_edits[3].text()
            if ccd_y_text: self.ccd[1] = float(ccd_y_text)
            
            dist_focal_text = line_edits[4].text()
            if dist_focal_text: self.dist_foc = float(dist_focal_text)
            
            skew_factor_text = line_edits[5].text()
            if skew_factor_text: self.stheta = float(skew_factor_text)
            
            #Recalcular parâmetros derivados
            self.ox = self.px_base / 2
            self.oy = self.px_altura / 2
            self.sx = self.px_base / self.ccd[0]
            self.sy = self.px_altura / self.ccd[1]
                
            print(f"""
                =========================================================
                Função: update_params_intrinsc
                =========================================================
                Novos Parâmetros Intrínsecos:
                - Resolução (w, h):    ({self.px_base}, {self.px_altura}) pixels
                - Sensor CCD (x, y):   {self.ccd} mm
                - Distância Focal:     {self.dist_foc}
                - Skew (s_theta):      {self.stheta}
                - Ponto Principal (ox, oy): ({self.ox}, {self.oy}) [px]
                - Skew x,y (sx,sy): ({self.sx:.2f}, {self.sy:.2f}) [px/mm]
                =========================================================
                """)
            
            #- Atualizar a visualização
            self.update_canvas()

        except ValueError as e:
            print(f"Erro ao fornecer dados de usuário no widget Parâmetros Intrínsecos: {e}")

    def update_world(self,line_edits):
        """
        Método responsável por atualizar a pose da câmera em relação ao referencial do MUNDO (pré-multiplicação).
        """
        try:
            dx = float(line_edits[0].text() or "0")
            angulo_x = float(line_edits[1].text() or "0")
            dy = float(line_edits[2].text() or "0")
            angulo_y = float(line_edits[3].text() or "0")
            dz = float(line_edits[4].text() or "0")
            angulo_z = float(line_edits[5].text() or "0")

            #Conversão dos ângulos de graus para radiano
            angulo_x_rad = (pi/180)*angulo_x
            angulo_y_rad = (pi/180)*angulo_y
            angulo_z_rad = (pi/180)*angulo_z

            #A ordem que será aplicado é T, Rz, Ry, Rx no referencial do mundo
            rx = Rx(angulo_x_rad)
            ry = Ry(angulo_y_rad)
            rz = Rz(angulo_z_rad)
            T = move(dx,dy,dz)
            
            #Pré-multiplicação para transformações no referencial do mundo
            self.cam = T @ rz @ ry @ rx @ self.cam

            print(f"""
                =========================================================
                Valores ATUALIZADOS (Função: update_world)
                =========================================================
                Transformação no referencial do MUNDO:
                - Translação (dx, dy, dz): ({dx:.2f}, {dy:.2f}, {dz:.2f})
                - Rotação (ax, ay, az):    ({angulo_x:.2f}, {angulo_y:.2f}, {angulo_z:.2f}) graus
                - Câmera Atual:\n{self.cam}
                =========================================================
            """)
            
            #Atualizar a visualização
            self.update_canvas()

        except ValueError as e:
            print(f"Erro ao fornecer dados de usuário no widget do Mundo: {e}")

    def update_cam(self,line_edits):
        """
        Método responsável por atualizar a pose da câmera em relação ao seu PRÓPRIO referencial (pós-multiplicação).
        """
        try:
            dx = float(line_edits[0].text() or "0")
            angulo_x = float(line_edits[1].text() or "0")
            dy = float(line_edits[2].text() or "0")
            angulo_y = float(line_edits[3].text() or "0")
            dz = float(line_edits[4].text() or "0")
            angulo_z = float(line_edits[5].text() or "0")
            
            #Conversão de graus para radianos
            angulo_x_rad = (pi/180)*angulo_x
            angulo_y_rad = (pi/180)*angulo_y
            angulo_z_rad = (pi/180)*angulo_z

            #Cria as matrizes de transformação local
            rx = Rx(angulo_x_rad)
            ry = Ry(angulo_y_rad)
            rz = Rz(angulo_z_rad)
            T = move(dx,dy,dz)

            #Pós-multiplicação para transformações no referencial da câmera
            # A ordem (T @ rz @ ry @ rx) define a sequência de operações locais
            transformacao_local = T @ rz @ ry @ rx
            self.cam = self.cam @ transformacao_local

            print(f"""
                =========================================================
                Valores ATUALIZADOS (Função: update_cam)
                =========================================================
                Transformação no referencial da CÂMERA:
                - Translação (dx, dy, dz): ({dx:.2f}, {dy:.2f}, {dz:.2f})
                - Rotação (ax, ay, az):    ({angulo_x:.2f}, {angulo_y:.2f}, {angulo_z:.2f}) graus
                - Câmera Atual:\n{self.cam}
                =========================================================
            """)
            
            #Atualizar a visualização
            self.update_canvas()

        except ValueError as e:
            print(f"Erro ao fornecer dados de usuário no widget da Câmera: {e}")

    def projection_2d(self):
        """
        Método que faz a projeção de um ponto 'p' no R^3 no R^2.
        """
        k = self.generate_intrinsic_params_matrix()
        # Matriz de transformação do mundo para a câmera
        T_cam_inv = np.linalg.inv(self.cam)
        
        # Projeção: K * [I|0] * T_cam_inv * P_mundo
        projecao_2d = k @ self.projection_matrix @ T_cam_inv @ self.objeto
        
        # Divisão perspectiva (ignora pontos com z_cam <= 0 para evitar divisão por zero/negativo)
        z_cam = projecao_2d[2,:]
        #Adicionado um valor pequeno para evitar divisão por zero
        z_cam[z_cam == 0] = 1e-6 
        
        projecao_2d = projecao_2d / z_cam
        
        return projecao_2d
    
    def generate_intrinsic_params_matrix(self):
        """
        Método responsável por gerar a matriz de parâmetros intrísicos da câmera.
        """
        k = np.array([
            [self.dist_foc * self.sx, self.dist_foc * self.stheta, self.ox],
            [0, self.dist_foc * self.sy, self.oy],
            [0, 0, 1]
        ])
        return k
    
    #Função para redesenhar os gráficos
    def update_canvas(self):
        """
        Limpa e redesenha ambos os gráficos (2D e 3D) com os valores atuais.
        """
        # Limpa os eixos
        self.ax1.cla()
        self.ax2.cla()

        # --- Gráfico 2D (Imagem) ---
        self.ax1.set_title("Imagem 2D Projetada")
        self.ax1.set_xlabel("x (pixels)")
        self.ax1.set_ylabel("y (pixels)")
        self.ax1.set_xlim([0, self.px_base])
        self.ax1.set_ylim([self.px_altura, 0]) # Eixo Y invertido para imagem
        self.ax1.set_aspect('equal', adjustable='box')
        self.ax1.grid(True)
        
        object_2d = self.projection_2d()
        self.ax1.plot(object_2d[0, :], object_2d[1, :], 'b.-')
        
        # --- Gráfico 3D (Mundo) ---
        self.ax2.set_title("Cena 3D")
        self.ax2.set_xlabel("X")
        self.ax2.set_ylabel("Y")
        self.ax2.set_zlabel("Z")
        lims = [-30, 60]
        self.ax2.set_xlim(lims)
        self.ax2.set_ylim(lims)
        self.ax2.set_zlim(lims)
        self.ax2.set_aspect('equal')

        # Desenha o referencial do Mundo (em (0,0,0))
        draw_arrows(self.world_frame[:, -1], self.world_frame[:, 0:-1], self.ax2, length=5)
        
        # Desenha o referencial da Câmera (na sua pose atual)
        draw_arrows(self.cam[:, -1], self.cam[:, 0:-1], self.ax2, length=10)

        # Desenha o objeto 3D
        self.ax2.plot(self.objeto[0,:], self.objeto[1,:], self.objeto[2,:], 'g-')

        # Redesenha os canvas
        self.canvas1.draw()
        self.canvas2.draw()
    
    #Função para o botão de reset
    def reset_canvas(self):
        """
        Reseta todas as variáveis e parâmetros para seus valores iniciais e atualiza a tela.
        """
        print("\n--- RESETANDO APLICAÇÃO ---")
        self.set_variables()
        self.update_canvas()

if __name__ == '__main__':
    from math import pi
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())