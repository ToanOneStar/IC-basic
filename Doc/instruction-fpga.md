# Giới thiệu cơ bản về FPGA
Đây là tổng hợp các kiến thức về cấu trúc, nguyên lý hoạt động và quy trình thiết kế FPGA, cùng với một số khái niệm liên quan trong lĩnh vực điện tử và máy tính.

### 1. Cấu trúc chính của FPGA
Một FPGA (Field-Programmable Gate Array) có ba thành phần cơ bản:
* **Các Khối Logic có thể Lập trình (CLBs):** Nơi thực hiện các chức năng logic. Mỗi CLB chứa:
    * **LUTs (Look-Up Tables):** Triển khai logic tổ hợp.
    * **Flip-Flops:** Lưu trữ trạng thái cho logic tuần tự.
* **Các Khối I/O có thể Lập trình:** Giao diện giữa FPGA và bên ngoài, hỗ trợ nhiều chuẩn tín hiệu (ví dụ: LVDS, LVTTL, CMOS).
* **Mạng Kết nối có thể Lập trình (Routing):** Hệ thống dây nối linh hoạt để kết nối các khối logic.

### 2. Các Khối Hỗ trợ Chuyên dụng
Ngoài các thành phần cơ bản, FPGA hiện đại còn có các khối chuyên dụng để tối ưu hóa hiệu suất:
* **Block RAM (BRAM):** Các khối bộ nhớ tích hợp sẵn, hiệu quả hơn việc tạo bộ nhớ bằng LUTs.
* **DSP Slices:** Các khối xử lý số học chuyên dụng (bộ nhân tích lũy) để tăng tốc các phép toán DSP, thường hiệu quả hơn LUTs.
* **PLL (Phase-Locked Loop):** Mạch tạo và đồng bộ tín hiệu xung nhịp, giúp nhân, chia hoặc làm sạch tín hiệu.

### 3. Phương pháp lập trình và Routing
Các đường kết nối trong FPGA được lập trình bằng các công nghệ khác nhau:
| Phương pháp | Đặc điểm | Ưu điểm | Nhược điểm |
| :--- | :--- | :--- | :--- |
| **SRAM** | Dựa trên các ô nhớ SRAM điều khiển công tắc. | Lập trình lại nhiều lần, tốc độ nhanh. | Dễ bay hơi (cần bộ nhớ ngoài), tiêu thụ năng lượng tĩnh. |
| **Antifuse** | Phá vỡ vật lý lớp cách điện bằng xung điện áp cao. | Lưu trữ vĩnh viễn, bảo mật cao, tiêu thụ ít năng lượng tĩnh. | Chỉ lập trình được một lần, không thể sửa đổi. |

Cả hai phương pháp đều sử dụng tệp cấu hình **bitstream** để nạp dữ liệu vào chip.

### 4. Quy trình thiết kế FPGA
Quá trình thiết kế một mạch trên FPGA gồm 5 bước chính:
1.  **Viết mô tả phần cứng:** Sử dụng ngôn ngữ **HDL** (VHDL/Verilog) để mô tả mạch.
2.  **Mô phỏng:** Dùng **testbench** để kiểm tra logic của mã HDL trên các công cụ mô phỏng như **ModelSim**.
3.  **Tổng hợp (Synthesis):** Chuyển đổi mã HDL thành một netlist (danh sách các khối logic).
4.  **Place & Route:** Sắp xếp và kết nối các khối logic trên chip.
5.  **Sinh bitstream và nạp:** Tạo tệp bitstream từ kết quả Place & Route và nạp vào FPGA.

### 5. So sánh với CPU và GPU
| | **FPGA** | **CPU** | **GPU** |
| :--- | :--- | :--- | :--- |
| **Bản chất** | Mạch phần cứng có thể lập trình | Bộ xử lý đa năng | Bộ xử lý song song |
| **Linh hoạt** | Rất cao (tùy biến phần cứng) | Rất cao (phần mềm) | Cao (lập trình song song) |
| **Ưu điểm** | Tối ưu hóa, độ trễ thấp, song song thực sự. | Đa năng, xử lý tuần tự tốt. | Xử lý song song cực mạnh, băng thông bộ nhớ cao. |
| **Ứng dụng** |	Xử lý video thời gian thực, viễn thông, hệ thống điều khiển.|	Máy tính cá nhân, máy chủ.|	Đồ họa, AI, tính toán hiệu năng cao.|

### 6. Các khái niệm đo lường hiệu suất
* **MIPS:** Million Instructions Per Second (Số triệu lệnh/giây).
* **FLOPS:** Floating-Point Operations Per Second (Số phép tính dấu phẩy động/giây).
* **TOPS:** Tera Operations Per Second (Số ngàn tỷ phép tính/giây), thường dùng cho AI.
* **IPC:** Instructions Per Cycle (Số lệnh/chu kỳ).
* **CUDA core:** Lõi xử lý song song cơ bản trong GPU của NVIDIA.
* **SIMD lanes:** Các đơn vị xử lý song song thực hiện cùng một lệnh trên nhiều dữ liệu.

### 7. Hãng sản xuất & Bo mạch
* **Các hãng lớn:** AMD (Xilinx), Intel (Altera), Lattice.
* **Hãng sản xuất bo mạch:** Digilent là một hãng phổ biến chuyên sản xuất các bo mạch phát triển cho FPGA.
* **Bo mạch nổi bật:** **Nexys Video** sử dụng chip **Artix-7** của Xilinx, chuyên dùng cho các ứng dụng xử lý hình ảnh và video.