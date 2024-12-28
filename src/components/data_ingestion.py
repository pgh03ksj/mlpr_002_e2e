import os  # Module để thao tác với hệ thống file và thư mục
import sys  # Module để truy cập các tham số và hàm của Python runtime
from src.exception import CustomException  # Import lớp exception tùy chỉnh
from src.logger import logging  # Import công cụ logging tùy chỉnh
import pandas as pd  # Thư viện xử lý dữ liệu dưới dạng DataFrame

from sklearn.model_selection import train_test_split  # Hàm để tách dữ liệu thành tập train/test
from dataclasses import dataclass  # Thư viện để tạo class cấu hình đơn giản

from src.components.data_transformation import DataTransformation  # Import lớp DataTransformation
from src.components.data_transformation import DataTransformationConfig  # Import cấu hình cho DataTransformation

from src.components.model_trainer import ModelTrainerConfig  # Import cấu hình cho ModelTrainer
from src.components.model_trainer import ModelTrainer  # Import lớp ModelTrainer

"""
Mô tả:
File này thực hiện các nhiệm vụ liên quan đến Data Ingestion (Thu thập dữ liệu đầu vào):
1. Đọc dữ liệu từ file CSV có sẵn.
2. Lưu trữ dữ liệu gốc (raw data) vào một thư mục định trước.
3. Chia dữ liệu thành hai tập: Train và Test.
4. Lưu hai tập dữ liệu này vào thư mục artifacts để sử dụng trong các bước tiếp theo.
"""

# Class cấu hình cho Data Ingestion
@dataclass
class DataIngestionConfig:
    """
    Lớp này định nghĩa cấu hình cho quá trình Data Ingestion, bao gồm các đường dẫn để lưu trữ:
    - train_data_path: Đường dẫn lưu tập train.
    - test_data_path: Đường dẫn lưu tập test.
    - raw_data_path: Đường dẫn lưu dữ liệu raw (gốc).
    """
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

# Class thực hiện Data Ingestion
class DataIngestion:
    """
    Lớp này chịu trách nhiệm thực hiện:
    - Đọc file dữ liệu gốc.
    - Chia dữ liệu thành các tập train/test.
    - Lưu trữ dữ liệu vào các file theo đường dẫn cấu hình.
    """
    def __init__(self):
        """
        Hàm khởi tạo, thiết lập cấu hình cho quá trình Data Ingestion.
        """
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Thực hiện quá trình Data Ingestion:
        - Đọc dữ liệu từ file CSV.
        - Chia dữ liệu thành tập train và test.
        - Lưu các tập này vào các file theo đường dẫn cấu hình.
        
        Returns:
        --------
        tuple: Đường dẫn tới file train và file test.
        """
        logging.info("Entered the data ingestion method or component")
        try:
            # Đọc dữ liệu từ file CSV
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info('Read the dataset as dataframe')

            # Tạo thư mục nếu chưa tồn tại để lưu file
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Lưu dữ liệu raw vào file
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            # Tách dữ liệu thành tập train và test
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Lưu tập train và test vào file
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            # Trả về đường dẫn của tập train và test
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            # Bắt lỗi và đưa ra thông báo lỗi thông qua CustomException
            raise CustomException(e, sys)

# Điểm bắt đầu của chương trình
if __name__ == "__main__":
    """
    Chương trình chính:
    1. Thực hiện Data Ingestion để chuẩn bị dữ liệu train và test.
    2. Gọi bước Data Transformation để xử lý dữ liệu đầu vào.
    3. Gọi bước Model Training để huấn luyện mô hình.
    """
    # Khởi tạo đối tượng DataIngestion
    obj = DataIngestion()
    # Thực hiện Data Ingestion và lấy đường dẫn tập train/test
    train_data, test_data = obj.initiate_data_ingestion()

    # Khởi tạo và thực hiện biến đổi dữ liệu
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    # Khởi tạo và thực hiện huấn luyện mô hình
    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))