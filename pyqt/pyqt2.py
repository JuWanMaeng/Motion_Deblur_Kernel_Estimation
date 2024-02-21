import sys,os
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QWidget,QInputDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QEvent, QSize
from PyQt5.QtGui import QImage, QColor,QMovie
from PIL import Image

class ImageApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.folder_name=None

    def initUI(self):
        self.setWindowTitle('Image Coordinate Viewer')

        # Create the central widget and layout
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        main_layout = QVBoxLayout(self.central_widget)

        # Create a QHBoxLayout for the main content
        content_layout = QHBoxLayout()

        # Left: Original image
        self.selected_image_label = QLabel(self)
        content_layout.addWidget(self.selected_image_label)

        # Create a QVBoxLayout for the images on the right
        images_layout = QVBoxLayout()

        # Top right: Unscaled image
        self.unscaled_image_label = QLabel(self)
        images_layout.addWidget(self.unscaled_image_label)

        # Label under unscaled image
        self.unscaled_text_label = QLabel("Original kernel Image", self)
        images_layout.addWidget(self.unscaled_text_label)

        # Bottom right: time step gif
        self.gif_label = QLabel(self)
        images_layout.addWidget(self.gif_label)

        # Label under scaled image
        self.gif_text_label = QLabel("Time step gif", self)
        images_layout.addWidget(self.gif_text_label)

        # Add the QVBoxLayout to the main QHBoxLayout
        content_layout.addLayout(images_layout)

        # Add the QHBoxLayout to the main QVBoxLayout
        main_layout.addLayout(content_layout)

        # Load button below the images
        self.load_button = QPushButton('Load Image', self)
        self.load_button.clicked.connect(self.loadImage)
        main_layout.addWidget(self.load_button)

    def loadImage(self):
        options = QFileDialog.Options()
        filepath, _ = QFileDialog.getOpenFileName(self, "Load Image", "", "Images (*.png *.xpm *.jpg *.bmp);;All Files (*)", options=options)
        if filepath:
            img_name=filepath.split('/')[-1]
            folder_name=img_name.split('.')[0]

            self.folder_name=folder_name

            pixmap = QPixmap(filepath)
            self.selected_image_label.setPixmap(pixmap)
            self.selected_image_label.installEventFilter(self)

    def eventFilter(self, source, event):
        if (source == self.selected_image_label) and (event.type() == QEvent.MouseButtonPress):
            i,j = self.getCoord(event.pos().y(), event.pos().x())

           
            
            region_size=16
            patch_size = 256
            task='debug'

            # find patch number
            col = 5
            patch_number = (i//patch_size) * col +  (j//patch_size)

            # find region number
            col = 256 // region_size

            # [ 0 <= i,j <= 255 ]
            i = i - (patch_size * (i//patch_size))
            j = j - (patch_size * (j//patch_size))
            sub_patch_number = (i // region_size) * col + (j // region_size)

            print(f'{i}row {j}col patch number:{patch_number} sub patch number{sub_patch_number}')

            if self.folder_name[0]=='i':
                self.folder_name=self.folder_name[4:]

            # /raid/joowan/pyqt/16_no_reg/00000/0_0.png
            image_path =f'/raid/joowan/pyqt/{task}/{self.folder_name}/{patch_number}_{sub_patch_number}.png'
            self.showCoordImage(image_path)
            return True
        return super().eventFilter(source, event)


    
    def getCoord(self, x, y):
        return x,y

    def showCoordImage(self, coord_image_path):
        # 이미지를 QImage 객체로 불러옵니다.
        original_qimage = QImage(coord_image_path)
        if original_qimage.isNull():
            print(f"Failed to load image from: {coord_image_path}")
            return

        folder_name,kernel_name = coord_image_path.split('/')[-2], coord_image_path.split('/')[-1]
        folder_name = folder_name + '.png' # 000009.png/
        kernel_name = kernel_name.split('.')[0] + '_kernel.png'  # 10_221_kernel.png
        
        time_step_list = list(range(990, -1, -10))  # [990, 980, 970, ..., 10, 0]
        mother_path = '/raid/joowan/debug/kernel_check'
        img_list=[]
        gif_path =os.path.join(mother_path,folder_name, '990') + '/' + kernel_name.split('.')[0] +'.gif'
        print(gif_path)

 
        for time_step in time_step_list:
            kernel_path = os.path.join(mother_path,folder_name, str(time_step), kernel_name)
            kerenl_image = Image.open(kernel_path)
            img_list.append(kerenl_image)

        img_list[0].save(gif_path,save_all=True, append_images = img_list[1:], duration=0.05, loop=1 )

        # load gif file
        self.gif_movie = QMovie(gif_path)
        if not self.gif_movie.isValid():
            print(f"Failed to load GIF from: {gif_path}")
            return
        
    
        self.gif_movie.setSpeed(200)     # Double the playback speed
        self.gif_movie.setScaledSize(QSize(300, 300))
        self.gif_label.setMovie(self.gif_movie)
        self.gif_movie.start() # Start the GIF animation

        unscaled_pixmap = QPixmap.fromImage(original_qimage).scaled(300, 300, Qt.KeepAspectRatio)
        self.unscaled_image_label.setPixmap(unscaled_pixmap)

        unscaled_pixmap = QPixmap.fromImage(original_qimage).scaled(300, 300, Qt.KeepAspectRatio)
        self.unscaled_image_label.setPixmap(unscaled_pixmap)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageApp()
    window.resize(800, 400)
    window.show()
    sys.exit(app.exec_())
