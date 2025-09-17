# Thiết kế CPU/GPU cho ứng dụng Mô hình Ngôn ngữ nhỏ (SLM)
## 1. Tìm hiểu về SLM
**Mô hình ngôn ngữ nhỏ (SLM)** là các mô hình AI có số lượng tham số ít hơn đáng kể so với các mô hình ngôn ngữ lớn (LLM). Thay vì được đào tạo trên lượng dữ liệu khổng lồ và thực hiện các tác vụ đa năng, SLM tập trung vào một hoặc một vài nhiệm vụ cụ thể.

### 1.1. Đặc điểm nổi bật

* **Kích thước**: SLM thường có vài tỷ tham số, trong khi LLM có thể lên đến hàng trăm tỷ hoặc thậm chí hàng nghìn tỷ.
* **Hiệu suất**: Do kích thước nhỏ gọn, SLM tiêu thụ ít tài nguyên tính toán và năng lượng hơn.
* **Tốc độ**: Tốc độ xử lý của SLM nhanh hơn, với độ trễ thấp hơn so với LLM.
* **Chi phí**: Chi phí đào tạo và vận hành thấp hơn rất nhiều, giúp các doanh nghiệp nhỏ và cá nhân dễ dàng tiếp cận hơn.
* **Bảo mật**: SLM có thể được triển khai và chạy trực tiếp trên các thiết bị cục bộ (như laptop hoặc điện thoại), không cần gửi dữ liệu lên đám mây, giúp tăng cường bảo mật và quyền riêng tư.

### 1.2. Ứng dụng phổ biến

SLM được sử dụng trong các lĩnh vực cần giải quyết một vấn đề cụ thể, ưu tiên tốc độ, hiệu quả và bảo mật.

* **Thiết bị di động và máy tính cá nhân**: Chạy các ứng dụng AI ngay trên thiết bị mà không cần kết nối internet, ví dụ như chatbot hoặc trợ lý ảo cá nhân.
* **Thiết bị biên (Edge computing)**: Tích hợp vào các thiết bị thông minh như robot, camera an ninh hoặc thiết bị IoT để xử lý dữ liệu ngay tại chỗ.
* **Hỗ trợ khách hàng**: Xây dựng các chatbot chuyên biệt cho từng doanh nghiệp để trả lời các câu hỏi thường gặp, phân tích cảm xúc của khách hàng.
* **Tóm tắt và phân loại văn bản**: Tóm tắt các email hoặc tài liệu dài, phân loại tài liệu theo chủ đề một cách nhanh chóng và hiệu quả.
* **Các ngành nghề chuyên biệt**: Trong y tế, SLM có thể được tinh chỉnh để phân tích dữ liệu bệnh án. Trong tài chính, có thể dùng để phân tích rủi ro thị trường.
* **Hỗ trợ lập trình**: Giúp hoàn thành code, gỡ lỗi hoặc tự động viết các đoạn code đơn giản.

## 2. Mục tiêu đề tài
Mục tiêu của đề tài này là **thiết kế và mô phỏng một bộ xử lý chuyên dụng (accelerator) được tối ưu hóa cho các mô hình ngôn ngữ nhỏ (SLM)**. Thay vì sử dụng CPU hoặc GPU đa năng, bộ xử lý này sẽ tập trung vào việc tăng tốc độ và hiệu quả năng lượng cho các tác vụ tính toán cốt lõi của SLM.

### 2.1. Mục tiêu chi tiết

* **Thiết kế kiến trúc phần cứng chuyên dụng**:
    * Tập trung vào các phép tính **đại số tuyến tính (linear algebra)**, bao gồm phép nhân ma trận-vector và ma trận-ma trận. Đây là nền tảng của các mạng nơ-ron như transformer, RNN, GRU và LSTM.
    * Tối ưu hóa các đơn vị xử lý (processing units) như Tensor Cores (tương tự như trên GPU của NVIDIA) hoặc các đơn vị xử lý ma trận tùy chỉnh.

* **Tối ưu hóa bộ nhớ và băng thông**:
    * Các mô hình ngôn ngữ lớn cần lượng bộ nhớ khổng lồ. Mục tiêu của đề tài là thiết kế kiến trúc bộ nhớ cache, bộ nhớ on-chip (SRAM) và giao tiếp bộ nhớ ngoài (DRAM) để giảm thiểu độ trễ và tối đa hóa băng thông.
    * Đặc biệt, tối ưu hóa các mô hình **attention, GRU và LSTM** bằng cách giảm số lần truy cập bộ nhớ và tăng hiệu quả sử dụng dữ liệu. Ví dụ, thiết kế các mô-đun phần cứng riêng biệt để xử lý phép tính attention.

* **Đảm bảo hiệu suất cao (low latency) khi inference**:
    * **Latency thấp** là một mục tiêu quan trọng, đặc biệt cho các ứng dụng thời gian thực như chatbot hoặc trợ lý ảo. Thiết kế phải đảm bảo tốc độ phản hồi nhanh, tức là thời gian từ khi nhận dữ liệu đầu vào đến khi trả về kết quả là ngắn nhất.
    * Mục tiêu này đạt được thông qua việc xử lý song song mạnh mẽ (massively parallel processing) và pipeline hóa các phép tính.

* **Nâng cao hiệu quả năng lượng (energy efficiency)**:
    * Bộ xử lý chuyên dụng phải tiêu thụ ít năng lượng hơn so với các giải pháp đa năng như GPU hoặc CPU khi thực hiện cùng một tác vụ.
    * Mục tiêu này đạt được bằng cách loại bỏ các thành phần không cần thiết cho SLM và sử dụng các kỹ thuật như **tính toán lượng tử hóa (quantization)**. Ví dụ, sử dụng các phép tính 8-bit hoặc 4-bit thay vì 16-bit hay 32-bit để giảm kích thước mô hình và lượng điện năng tiêu thụ.

### 2.2. Tầm quan trọng của đề tài

Đề tài này có ý nghĩa lớn trong bối cảnh AI đang trở nên phổ biến. Thay vì chỉ phụ thuộc vào các trung tâm dữ liệu lớn với các siêu máy tính, việc phát triển các bộ xử lý chuyên dụng sẽ giúp:

* **Tăng tốc độ xử lý tại chỗ (on-device processing)**: Cho phép các thiết bị nhỏ gọn như điện thoại, laptop, hoặc các thiết bị IoT có thể chạy SLM một cách hiệu quả.
* **Tiết kiệm chi phí**: Giảm chi phí vận hành và bảo trì cho các doanh nghiệp.
* **Bảo mật dữ liệu**: Dữ liệu được xử lý cục bộ, không cần phải gửi lên đám mây, giúp bảo vệ quyền riêng tư.

### 2.3. Các thuật ngữ liên quan
**Attention, GRU, và LSTM** là ba kiến trúc mạng nơ-ron được sử dụng phổ biến trong xử lý ngôn ngữ tự nhiên (NLP) và các bài toán xử lý chuỗi dữ liệu khác. Chúng được thiết kế để giải quyết những hạn chế của các mạng nơ-ron truyền thống, đặc biệt là khả năng ghi nhớ thông tin dài hạn.

#### 2.3.1 Mạng LSTM (Long Short-Term Memory)

**LSTM** là một loại mạng nơ-ron hồi quy (RNN) được phát triển để giải quyết vấn đề "gradient vanishing" trong RNN. Vấn đề này khiến mạng không thể học và ghi nhớ các phụ thuộc dài hạn trong một chuỗi dữ liệu.

Mỗi tế bào LSTM có cấu trúc phức tạp hơn một tế bào RNN thông thường, bao gồm các "cổng" (gates) điều khiển dòng thông tin:

* **Cổng quên (Forget gate):** Quyết định thông tin nào từ trạng thái tế bào trước đó cần được "quên" đi.
* **Cổng đầu vào (Input gate):** Quyết định thông tin mới nào từ đầu vào hiện tại cần được "nhớ" thêm vào trạng thái tế bào.
* **Cổng đầu ra (Output gate):** Quyết định thông tin nào từ trạng thái tế bào hiện tại sẽ được truyền ra ngoài dưới dạng đầu ra.

Nhờ cơ chế này, LSTM có thể ghi nhớ thông tin qua nhiều bước thời gian một cách hiệu quả, làm cho nó trở nên mạnh mẽ trong các tác vụ như dịch máy, tạo văn bản và nhận dạng giọng nói.


#### 2.3.2. Mạng GRU (Gated Recurrent Unit)

**GRU** là một phiên bản đơn giản hơn của LSTM, được giới thiệu với mục tiêu giảm độ phức tạp tính toán nhưng vẫn duy trì hiệu quả. Nó kết hợp hai trong ba cổng của LSTM thành một đơn vị duy nhất:

* **Cổng cập nhật (Update gate):** Điều khiển lượng thông tin từ trạng thái trước đó cần được giữ lại và lượng thông tin mới từ đầu vào hiện tại cần được thêm vào. Cổng này đóng vai trò tương tự như cổng quên và cổng đầu vào của LSTM.
* **Cổng đặt lại (Reset gate):** Quyết định lượng thông tin từ trạng thái ẩn trước đó sẽ được "đặt lại" (quên đi) khi tính toán trạng thái ẩn mới.

Nhờ kiến trúc tinh giản, GRU có tốc độ tính toán nhanh hơn và ít tham số hơn so với LSTM, nhưng hiệu suất trên nhiều bài toán lại tương đương. Vì vậy, GRU thường là lựa chọn ưu tiên khi tài nguyên tính toán bị hạn chế.

#### 2.3.3. Cơ chế Attention

**Attention** là một **cơ chế** cho phép mô hình tập trung vào các phần quan trọng của dữ liệu đầu vào khi tạo ra đầu ra. 

Trong các mô hình dịch máy, thay vì nén toàn bộ thông tin của câu đầu vào thành một vector duy nhất, cơ chế Attention cho phép mô hình xem xét lại toàn bộ câu gốc ở mỗi bước tạo từ trong câu đầu ra. Nó tính toán một "trọng số" (weight) cho mỗi từ đầu vào, biểu thị mức độ quan trọng của từ đó đối với việc tạo ra từ đầu ra hiện tại.


Cơ chế Attention đã thay đổi hoàn toàn lĩnh vực NLP và là thành phần cốt lõi của kiến trúc **Transformer**, nền tảng của các mô hình ngôn ngữ lớn (LLM) hiện đại như GPT-3 và BERT. Khác với LSTM và GRU, Transformer sử dụng Attention thay vì kiến trúc tuần tự, cho phép xử lý song song và tăng tốc độ đào tạo đáng kể.

#### 2.3.4. Domain-specific accelerator
**Domain-specific accelerator** là một con chip hoặc bộ xử lý được thiết kế đặc biệt để tăng tốc độ và hiệu quả năng lượng cho các tác vụ tính toán cụ thể trong một lĩnh vực (domain) nhất định. Thay vì được thiết kế để xử lý đa nhiệm như CPU hoặc GPU, chúng được tối ưu hóa cho một loại thuật toán hoặc ứng dụng cụ thể.
