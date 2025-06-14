import sys
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QGridLayout, QLabel, QWidget, QLineEdit, QHBoxLayout, QVBoxLayout, QPushButton,QGroupBox
from PyQt5.QtGui import QDoubleValidator
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D
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

    def set_variables(self):
        self.objeto_original = [] #modificar
        self.objeto = self.objeto_original
        self.cam_original = np.hstack((Base(),np.array([[0,0,0,1]]).T))#Câmera na origem
        self.cam = self.cam_original
        self.px_base = 1280  #modificar
        self.px_altura = 720 #modificar
        self.dist_foc = 50 #modificar
        self.stheta = 0 #modificar
        self.ox = self.px_base/2 #modificar
        self.oy = self.px_altura/2 #modificar
        self.ccd = [36,24] #modificar
        self.projection_matrix = [] #modificar
        self.sx = self.px_base/self.ccd[0]
        self.sy = self.px_altura/self.ccd[1]

        print(f"""
            ======================================================
            Função: set_variables
            ======================================================
            Parâmetros de Geometria:
            - Objeto Original:     {self.objeto_original}
            - Objeto Atual:        {self.objeto}
            - Câmera Original:     {self.cam_original}
            - Câmera Atual:        {self.cam}

            Parâmetros Intrínsecos da Câmera:
            - Resolução (w, h):    ({self.px_base}, {self.px_altura}) [px]
            - Ponto Principal (ox, oy): ({self.ox}, {self.oy}) [px]
            - Sensor CCD (x, y):   {self.ccd} [mm]
            - Distância Focal:     {self.dist_foc}
            - Skew (s_theta):      {self.stheta}
            - Skew x,y (sx,sy): {self.sx:.2f},{self.sy:.2f} [px]/[mm]

            Matrizes:
            - Matriz de Projeção:  {self.projection_matrix}
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
        reset_button.setFixedSize(50, 30)  # Define um tamanho fixo para o botão (largura: 50 pixels, altura: 30 pixels)
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
            validator = QDoubleValidator()  # Validador numérico
            line_edit.setValidator(validator)  # Aplicar o validador ao QLineEdit
            grid_layout.addWidget(label, (i-1)//2, 2*((i-1)%2))
            grid_layout.addWidget(line_edit, (i-1)//2, 2*((i-1)%2) + 1)
            line_edits.append(line_edit)

        # Criar o botão de atualização
        update_button = QPushButton("Atualizar")

        ##### Você deverá criar, no espaço reservado ao final, a função self.update_params_intrinsc ou outra que você queira 
        # Conectar a função de atualização aos sinais de clique do botão
        update_button.clicked.connect(lambda: self.update_params_intrinsc(line_edits))

        # Adicionar os widgets ao layout do widget line_edit_widget
        line_edit_layout.addLayout(grid_layout)
        line_edit_layout.addWidget(update_button)

        # Retornar o widget e a lista de caixas de texto
        return line_edit_widget
    
    def create_world_widget(self, title):
        # Criar um widget para agrupar os QLineEdit
        line_edit_widget = QGroupBox(title)
        line_edit_layout = QVBoxLayout()
        line_edit_widget.setLayout(line_edit_layout)

        # Criar um layout de grade para dividir os QLineEdit em 3 colunas
        grid_layout = QGridLayout()

        line_edits = []
        labels = ['X(move):', 'X(angle):', 'Y(move):', 'Y(angle):', 'Z(move):', 'Z(angle):']  # Texto a ser exibido antes de cada QLineEdit

        # Adicionar widgets QLineEdit com caixa de texto ao layout de grade
        for i in range(1, 7):
            line_edit = QLineEdit()
            label = QLabel(labels[i-1])
            validator = QDoubleValidator()  # Validador numérico
            line_edit.setValidator(validator)  # Aplicar o validador ao QLineEdit
            grid_layout.addWidget(label, (i-1)//2, 2*((i-1)%2))
            grid_layout.addWidget(line_edit, (i-1)//2, 2*((i-1)%2) + 1)
            line_edits.append(line_edit)

        # Criar o botão de atualização
        update_button = QPushButton("Atualizar")

        ##### Você deverá criar, no espaço reservado ao final, a função self.update_world ou outra que você queira 
        # Conectar a função de atualização aos sinais de clique do botão
        update_button.clicked.connect(lambda: self.update_world(line_edits))

        # Adicionar os widgets ao layout do widget line_edit_widget
        line_edit_layout.addLayout(grid_layout)
        line_edit_layout.addWidget(update_button)

        # Retornar o widget e a lista de caixas de texto
        return line_edit_widget

    def create_cam_widget(self, title):
        # Criar um widget para agrupar os QLineEdit
        line_edit_widget = QGroupBox(title)
        line_edit_layout = QVBoxLayout()
        line_edit_widget.setLayout(line_edit_layout)

        # Criar um layout de grade para dividir os QLineEdit em 3 colunas
        grid_layout = QGridLayout()

        line_edits = []
        labels = ['X(move):', 'X(angle):', 'Y(move):', 'Y(angle):', 'Z(move):', 'Z(angle):']  # Texto a ser exibido antes de cada QLineEdit

        # Adicionar widgets QLineEdit com caixa de texto ao layout de grade
        for i in range(1, 7):
            line_edit = QLineEdit()
            label = QLabel(labels[i-1])
            validator = QDoubleValidator()  # Validador numérico
            line_edit.setValidator(validator)  # Aplicar o validador ao QLineEdit
            grid_layout.addWidget(label, (i-1)//2, 2*((i-1)%2))
            grid_layout.addWidget(line_edit, (i-1)//2, 2*((i-1)%2) + 1)
            line_edits.append(line_edit)

        # Criar o botão de atualização
        update_button = QPushButton("Atualizar")

        ##### Você deverá criar, no espaço reservado ao final, a função self.update_cam ou outra que você queira 
        # Conectar a função de atualização aos sinais de clique do botão
        update_button.clicked.connect(lambda: self.update_cam(line_edits))

        # Adicionar os widgets ao layout do widget line_edit_widget
        line_edit_layout.addLayout(grid_layout)
        line_edit_layout.addWidget(update_button)

        # Retornar o widget e a lista de caixas de texto
        return line_edit_widget

    def create_matplotlib_canvas(self):
        # Criar um widget para exibir os gráficos do Matplotlib
        canvas_widget = QWidget()
        canvas_layout = QHBoxLayout()
        canvas_widget.setLayout(canvas_layout)

        # Criar um objeto FigureCanvas para exibir o gráfico 2D
        self.fig1, self.ax1 = plt.subplots()
        self.ax1.set_title("Imagem")
        self.canvas1 = FigureCanvas(self.fig1)

        ##### Falta acertar os limites do eixo X
        
        ##### Falta acertar os limites do eixo Y
        
        ##### Você deverá criar a função de projeção 
        object_2d = self.projection_2d()

        ##### Falta plotar o object_2d que retornou da projeção
          
        self.ax1.grid('True')
        self.ax1.set_aspect('equal')  
        canvas_layout.addWidget(self.canvas1)

        # Criar um objeto FigureCanvas para exibir o gráfico 3D
        self.fig2 = plt.figure()
        self.ax2 = self.fig2.add_subplot(111, projection='3d')
        
        ##### Falta plotar o seu objeto 3D e os referenciais da câmera e do mundo
        
        self.canvas2 = FigureCanvas(self.fig2)
        canvas_layout.addWidget(self.canvas2)

        # Retornar o widget de canvas
        return canvas_widget


    ##### Você deverá criar as suas funções aqui
    
    def update_params_intrinsc(self, line_edits):
        """
        Método Responsável para alterar os parâmetros intrísicos fornecido pelo usuario
        args:
            self: Referencia a classe.
            line_edits: Lista com cada um dos QLineEdit(inputs do usuário) do widget params instr(Ex: n_pixels_base,n_pixels_altura...)
        """
        try:
            #n_pixels_base ---------------------------------------------------------------------
            n_pixels_base = line_edits[0].text()
            if n_pixels_base:
                self.px_base = float(n_pixels_base)         
            #n_pixels_altura--------------------------------------------------------------------
            n_pixel_altura = line_edits[1].text()
            if n_pixel_altura:#Caso não seja vazio
                self.px_altura = float(n_pixel_altura)
            #ccd_x,ccd_y------------------------------------------------------------------------
            ccd_x = line_edits[2].text()
            ccd_y = line_edits[3].text()
            if ccd_x and ccd_y:
                self.ccd = [float(ccd_x),float(ccd_y)]
            #dist_focal-------------------------------------------------------------------------
            dist_focal = line_edits[4].text()
            if dist_focal:
                self.dist_foc = float(dist_focal)
            #skew factor------------------------------------------------------------------------
            skew_factor = line_edits[5].text()
            if skew_factor:
                self.stheta = skew_factor
            print(f"""
                =========================================================
                Função: update_params_intrinsc
                =========================================================
                Novos Parâmetros Intrínsecos:
                - Resolução (w, h):    ({self.px_base}, {self.px_altura}) pixels
                - Sensor CCD (x, y):   {self.ccd} mm
                - Distância Focal:     {self.dist_foc}
                - Skew (s_theta):      {self.stheta}
                =========================================================
                """)
                
                
            
        except ValueError:
            print(f"Error ao fornecer dados de usuário no widget Parâmetros Intrísicos:{ValueError}")
        return 

    def update_world(self,line_edits):
        """
        Método responsável por atualizar os valores que foram digitados pelo usuáriuo no QlineEdit(Input) para as variáveis no ref do mundo.
        args:
            self: Referência a classe.
            line_edits: Array com os QlineEdits.
        """
        try:
            # Converte o texto para float. Se o campo estiver vazio, usa 0.0 como padrão.
            dx = float(line_edits[0].text() or "0")
            angulo_x = float(line_edits[1].text() or "0")
            dy = float(line_edits[2].text() or "0")
            angulo_y = float(line_edits[3].text() or "0")
            dz = float(line_edits[4].text() or "0")
            angulo_z = float(line_edits[5].text() or "0")

            #A ordem que será aplicado é Rx,Ry,Rz e depois a matriz T
            rx = Rx(angulo_x)
            ry = Ry(angulo_y)
            rz = Rz(angulo_z)
            T=move(dx,dy,dz)
            self.cam = T@rz@ry@rx

            print(f"""
                =========================================================
                Valores ATUALIZADOS (Função: update_world)
                =========================================================
                Novos Parâmetros do Referencial do Mundo:
                - Translação (dx, dy, dz): ({dx:.2f}, {dy:.2f}, {dz:.2f})
                - Rotação (ax, ay, az):    ({angulo_x:.2f}, {angulo_y:.2f}, {angulo_z:.2f}) graus
                - Câmera Atual:{self.cam}
                =========================================================
            """)

        except ValueError:
            print(f"Error ao fornecer dados de usuário no widget para alterar os valores de Translação/Rotação do frame do Mundo: {ValueError}")
        return

    def update_cam(self,line_edits):
        """
        Método responsável por atualizar os valores que foram digitados pelo usuáriuo no QlineEdit(Input) para as variáveis no ref na câmera.
        args:
            self: Referência a classe.
            line_edits: Array com os QlineEdits.
        """
        try:
            # Converte o texto para float. Se o campo estiver vazio, usa 0.0 como padrão.
            dx = float(line_edits[0].text() or "0")
            angulo_x = float(line_edits[1].text() or "0")
            dy = float(line_edits[2].text() or "0")
            angulo_y = float(line_edits[3].text() or "0")
            dz = float(line_edits[4].text() or "0")
            angulo_z = float(line_edits[5].text() or "0")

            #A ordem que será aplicado é Rx,Ry,Rz e depois a matriz T-------------------------------
            rx = Rx(angulo_x)
            ry = Ry(angulo_y)
            rz = Rz(angulo_z)
            T=move(dx,dy,dz)
            #Rotação Rx entorno da própria câmera
            self.cam = self.cam@rx@self.cam_original
            #Rotação Ry entorno da própria câmera
            self.cam = self.cam@ry@self.cam_original
            #Rotação Rz entorno da própria câmera
            self.cam = self.cam@rz@self.cam_original
            #Translação entorno da câmera
            self.cam = self.cam@T@self.cam_original

            print(f"""
                =========================================================
                Valores ATUALIZADOS (Função: update_cam)
                =========================================================
                Novos Parâmetros do Referencial da Câmera:
                - Translação (dx, dy, dz): ({dx:.2f}, {dy:.2f}, {dz:.2f})
                - Rotação (ax, ay, az):    ({angulo_x:.2f}, {angulo_y:.2f}, {angulo_z:.2f}) graus
                - Câmera Atual:{self.cam}
                =========================================================
            """)

        except ValueError:
            print(f"Error ao fornecer dados de usuário no widget para alterar os valores de Translação/Rotação do frame do Mundo: {ValueError}")

        return 
    
    def projection_2d(self):
        """
        Método que faz a projeção de um ponto 'p' no R^3 no R^2.
        args:
            self: Referência a classe.
        """
        proj_canonica = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
        try:
            self.projection_matrix = self.generate_intrinsic_params_matrix()@proj_canonica@np.linalg.inv(self.cam)@self.objeto
            self.projection_matrix = self.projection_matrix/self.projection_matrix[2]
            print(f"""
                =========================================================
                Valores ATUALIZADOS (Função: projection_2d)
                =========================================================
                Matriz de projeção 2d:
                - Mp : {self.projection_matrix}
                =========================================================
            """)
        except ValueError:
            print(f'Error ao gerar a matriz de projeção 2d de um ponto 3d no mundo para 2d na câmera.')
        return 
    
    def generate_intrinsic_params_matrix(self):
        """
        Método responsável por gerar a matriz de parâmetros intrísicos da câmera.
        args:
            self: Referência a classe.
        """
        k = np.array([[self.dist_foc*self.sx,self.dist_foc*self.stheta,self.ox],
                         [0, self.dist_foc*self.sy,self.oy],
                         [0,0,1]
                         ])
        print(f"""
                =========================================================
                Valores ATUALIZADOS (Função: generate_intrinsic_params_matrix)
                =========================================================
                Matriz de parâmetros intrísicos:
                - K : {k}
                =========================================================
            """)
        return  k
    
        
    

    def update_canvas(self):
        return 
    
    def reset_canvas(self):
        return
    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
