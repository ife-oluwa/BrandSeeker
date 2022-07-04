from genericpath import exists
from borb.pdf import Document
from borb.pdf.page.page import Page
from borb.pdf.canvas.layout.page_layout.multi_column_layout import SingleColumnLayout
from borb.pdf.canvas.layout.page_layout.page_layout import PageLayout
from borb.pdf.pdf import PDF
from borb.pdf.canvas.layout.image.barcode import Barcode, BarcodeType  
from borb.pdf.canvas.layout.layout_element import LayoutElement
from borb.pdf.canvas.layout.text.paragraph import Paragraph  
from borb.pdf.canvas.color.color import HexColor  
from decimal import Decimal  
from borb.pdf.canvas.layout.table.flexible_column_width_table import FlexibleColumnWidthTable
from borb.pdf.canvas.layout.image.image import Image
from borb.pdf.canvas.layout.table.table import TableCell  
from borb.pdf.canvas.layout.table.fixed_column_width_table import FixedColumnWidthTable
from borb.pdf.canvas.layout.list.unordered_list import UnorderedList
from pathlib import Path
from borb.pdf.canvas.geometry.rectangle import Rectangle
from borb.pdf.canvas.layout.shape.shape import Shape
from borb.pdf.page.page_size import PageSize
from borb.pdf.canvas.line_art.line_art_factory import LineArtFactory
from datetime import date
import cv2
import numpy as np
import typing
import random
import shutil
import os


def add_gray_artwork_upper_right_corner(page: Page):
    """
    This method will add a gray artwork of squares and triangles in the upper right corner
    of the given Page
    """
    grays: typing.List[HexColor] = [
        HexColor("A9A9A9"),
        HexColor("D3D3D3"),
        HexColor("DCDCDC"),
        HexColor("E0E0E0"),
        HexColor("E8E8E8"),
        HexColor("F0F0F0"),
    ]
    ps: typing.Tuple[Decimal, Decimal] = PageSize.A4_PORTRAIT.value
    N: int = 4
    M: Decimal = Decimal(32)
    
    # Draw triangles
    for i in range(0, N):
        x: Decimal = ps[0] - N * M + i * M
        y: Decimal = ps[1] - (i + 1) * M
        rg: HexColor = random.choice(grays)
        Shape(
            points=[(x + M, y), (x + M, y + M), (x, y + M)],
            stroke_color=rg,
            fill_color=rg,
        ).layout(page, Rectangle(x, y, M, M))
        
    # Draw squares
    for i in range(0, N - 1):
        for j in range(0, N - 1):
            if j > i:
                continue
            x: Decimal = ps[0] - (N - 1) * M + i * M
            y: Decimal = ps[1] - (j + 1) * M
            rg: HexColor = random.choice(grays)
            Shape(
                points=[(x, y), (x + M, y), (x + M, y + M), (x, y + M)],
                stroke_color=rg,
                fill_color=rg,
            ).layout(page, Rectangle(x, y, M, M))

def add_colored_artwork_bottom_right_corner(page: Page):
    """
    This method will add a blue/purple artwork of lines 
    and squares to the bottom right corner
    of the given Page
    """
    ps: typing.Tuple[Decimal, Decimal] = PageSize.A4_PORTRAIT.value
    
    # Square
    Shape(
      points=[
          (ps[0] - 32, 40),
          (ps[0], 40),
          (ps[0], 40 + 32),
          (ps[0] - 32, 40 + 32),
      ],
      stroke_color=HexColor("d53067"),
      fill_color=HexColor("d53067"),
    ).layout(page, Rectangle(ps[0] - 32, 40, 32, 32))
    
    # Square
    Shape(
      points=[
          (ps[0] - 64, 40),
          (ps[0] - 32, 40),
          (ps[0] - 32, 40 + 32),
          (ps[0] - 64, 40 + 32),
      ],
      stroke_color=HexColor("eb3f79"),
      fill_color=HexColor("eb3f79"),
    ).layout(page, Rectangle(ps[0] - 64, 40, 32, 32))
    
    # Triangle
    Shape(
      points=[
          (ps[0] - 96, 40),
          (ps[0] - 64, 40),
          (ps[0] - 64, 40 + 32),
      ],
      stroke_color=HexColor("e01b84"),
      fill_color=HexColor("e01b84"),
    ).layout(page, Rectangle(ps[0] - 96, 40, 32, 32))
        
    # Line
    r: Rectangle = Rectangle(Decimal(0), Decimal(32), ps[0], Decimal(8))
    Shape(
      points=LineArtFactory.rectangle(r),
      stroke_color=HexColor("283592"),
      fill_color=HexColor("283592"),
    ).layout(page, r)

def load_frames(video_path, output_dict):
    
    tmp_path = 'img_tmp'
    try:
        shutil.rmtree(tmp_path)
    except OSError as e:
        print("Error: %s : %s" % (tmp_path, e.strerror))
    
    os.makedirs('img_tmp', exist_ok=True)
    
    # Sample for test
    TEST_DICT = {'Displate': [0.17413793299531197,
    0.7290941774845123,
    6728,
    [164.1377716064453,
    189.26400756835938,
    478.53680419921875,
    295.28814697265625]],
    'GFuel': [0.21769244941742322,
    0.616112232208252,
    4765,
    [174.75082397460938,
    102.81185150146484,
    343.7975769042969,
    163.09121704101562]],
    'Winamax': [0.09757052368013214,
    0.41104447841644287,
    447,
    [83.57308197021484, 56.4051399230957, 117.9732437133789, 79.14286041259766]]}

    #Collect the right images
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    length = frame_count/fps

    frame_seq_list = []
    brand_list = []
    confidence_list = []
    bbx = []
        
    for i in output_dict:
        # brand
        brand_list.append(i)
        # median confidence
        confidence_list.append(output_dict[i][1])
        # frame
        frame_seq_list.append(output_dict[i][2])
        # bbx
        bbx.append(output_dict[i][3])

    path_list = []
    path_cropped_list= []
    
    for i in range(0, len(frame_seq_list)):
        cap.set(1, frame_seq_list[i])  # Where frame_no is the frame you want
        ret, frame = cap.read()  # Read the frame
        if ret:
            cropped_frame = frame[int(bbx[i][1]):int(bbx[i][3]), int(bbx[i][0]):int(bbx[i][2])]
            cv2.imwrite('img_tmp/frame_'+str(frame_seq_list[i])+'.jpg', frame)
            cv2.imwrite('img_tmp/frame_'+str(frame_seq_list[i])+'_cropped.jpg', cropped_frame)
            path_list.append('img_tmp/frame_'+str(frame_seq_list[i])+'.jpg')
            path_cropped_list.append('img_tmp/frame_'+str(frame_seq_list[i])+'_cropped.jpg')
    
    cap.release()
    video_name = video_path.split(".")[0]

    return video_name, length, confidence_list, brand_list, path_list, path_cropped_list, frame_seq_list

def pdf_generator(video_path, output_dict):
    
    # Write frames and return a path list
    video_name, length, confidence_list, brand_list, path_list, path_cropped_list, frame_seq_list = load_frames(video_path, output_dict)
    minutes = int(length/60)
    seconds = length%60
    
    # Create empty Document
    pdf = Document()

    # Create empty Page
    page = Page()

    # Add Page to Document
    pdf.add_page(page)

    # Create PageLayout
    layout: PageLayout = SingleColumnLayout(page)

    layout.add(
        Paragraph("Brand Seeker", 
            font_color=HexColor("#6d64e8"), 
            font_size=Decimal(20)
        )
    )

    qr_code: LayoutElement = Barcode(
        data="https://github.com/HumanBojack/BrandSeeker",
        width=Decimal(64),
        height=Decimal(64),
        type=BarcodeType.QR,
    )
    
    layout.add(
        FlexibleColumnWidthTable(number_of_columns=2, number_of_rows=1)
        .add(qr_code)
        .add(
            Paragraph(
                """
                50 Rue de la gamelle
                Brand Seeker CA
                59000 FR
                """,
                padding_top=Decimal(12),
                respect_newlines_in_text=True,
                font_color=HexColor("#666666"),
                font_size=Decimal(10),
            )
        )
        .no_borders()
    )
    # Title
    layout.add(
        Paragraph(
            str(video_name), font_color=HexColor("#283592"), font_size=Decimal(34)
        )
    )
    # Subtitle
    layout.add(
        Paragraph(
            "Date: "+str(date.today().strftime("%B %d, %Y")),
            font_color=HexColor("#e01b84"),
            font_size=Decimal(11),
        )
    )
    layout.add(
        Paragraph(
            "Duration: "+ str(minutes) + 'm ' + str(int(seconds)) + 's',
            font_color=HexColor("#e01b84"),
            font_size=Decimal(11),
        )
    )
    for i in range(0, len(path_list)):
        layout.add(
        FixedColumnWidthTable(
            number_of_rows=2,
            number_of_columns=2,
            column_widths=[Decimal(0.3), Decimal(0.7)],
        )
        .add(
            # Add the frame
            TableCell(
                Image(
                    Path(path_list[i]),
                    width=Decimal(128),
                    height=Decimal(128),
                ),
                row_span=2,
            )
        )
        .add(
            Paragraph(
                brand_list[i],
                font_color=HexColor("e01b84"),
                font="Helvetica-Bold",
                padding_bottom=Decimal(10),
            )
        )
        .add(
            UnorderedList()
            .add(Paragraph("Frame " + str(frame_seq_list[i])))
            .add(Paragraph("Median confidence: " + str(confidence_list[i])))
            .add(Paragraph("Detected bounding box: "))
            .add(Image(Path(path_cropped_list[i],                    
                        width=Decimal(16),
                        height=Decimal(16))))
        )
        .no_borders()
    )

    for i in range(0, int(pdf.get_document_info().get_number_of_pages())):
        add_colored_artwork_bottom_right_corner(pdf.get_page(i))
        add_gray_artwork_upper_right_corner(pdf.get_page(i))

    with open("output.pdf", "wb") as pdf_file_handle:
        PDF.dumps(pdf_file_handle, pdf)
