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

## 3. Kiến trúc hệ thống

### 3.1. CPU Controller (Khối điều khiển trung tâm)

**CPU Controller** là bộ não quản lý toàn bộ hoạt động của accelerator. Nó không thực hiện các phép toán phức tạp mà tập trung vào nhiệm vụ điều phối và quản lý. 

* **Nhiệm vụ chính**:
    * **Điều khiển pipeline và luồng lệnh**: Quyết định lệnh nào sẽ được thực thi, khi nào và trên khối nào. Giống như một người quản lý dự án, nó đảm bảo các tác vụ được hoàn thành theo đúng trình tự và thời gian.
    * **Quản lý lệnh `load/store`**: Khi mô hình cần dữ liệu (ví dụ: các ma trận trọng số) từ RAM, CPU Controller sẽ ra lệnh cho DMA Engine để nạp dữ liệu vào. Sau khi tính toán xong, nó cũng ra lệnh để lưu kết quả trở lại RAM.
    * **Điều khiển tổng thể**: Khởi động, tạm dừng, và kết thúc quá trình xử lý, đồng thời xử lý các ngoại lệ hoặc lỗi phát sinh.

### 3.2. DMA Engine (Công cụ truy cập bộ nhớ trực tiếp)

**DMA Engine** là một khối chuyên dụng được thiết kế để di chuyển dữ liệu một cách hiệu quả giữa RAM và bộ nhớ trên chip (on-chip memory).

* **Nhiệm vụ chính**:
    * **Truyền dữ liệu không cần can thiệp của CPU**: Đây là điểm mấu chốt. CPU Controller chỉ cần ra lệnh một lần duy nhất cho DMA Engine về việc cần di chuyển dữ liệu nào. Sau đó, DMA Engine sẽ tự động thực hiện việc này một cách độc lập, giải phóng CPU Controller để nó có thể làm các công việc khác.
    * **Tăng tốc độ truyền tải**: Do được tối ưu hóa cho nhiệm vụ này, DMA Engine có thể truyền một lượng lớn dữ liệu (batch data) như đầu vào của mô hình, ma trận trọng số, hoặc các kết quả trung gian nhanh hơn nhiều so với việc để CPU tự thực hiện. 

### 3.3. Memory Hierarchy (Bộ nhớ phân cấp)

**Memory Hierarchy** là một hệ thống các loại bộ nhớ khác nhau được tổ chức theo cấp độ để đảm bảo dữ liệu luôn có sẵn cho khối tính toán một cách nhanh nhất có thể.

* **Nhiệm vụ chính**:
    * **SRAM On-chip**: Loại bộ nhớ cực nhanh nhưng đắt đỏ và có dung lượng nhỏ, nằm ngay trên con chip. Nó được sử dụng để lưu trữ các dữ liệu "nóng" (hot data), tức là những dữ liệu thường xuyên được sử dụng, như các ma trận trọng số của lớp mạng đang được tính toán.
    * **Cache**: Các cache nhỏ hơn (L1, L2) được đặt gần các đơn vị tính toán để lưu tạm thời những dữ liệu vừa được sử dụng, giúp truy cập lại ngay lập tức mà không cần đi xa hơn đến SRAM hay RAM.
    * **Đảm bảo băng thông và giảm độ trễ**: Mục tiêu của hệ thống này là giảm thiểu số lần phải truy cập vào RAM chậm, vốn là nút thắt cổ chai lớn nhất trong các hệ thống tính toán.

### 3.4. Accelerator

**Accelerator** là trái tim của hệ thống, nơi tất cả các phép tính phức tạp của mô hình SLM được thực hiện.

* **Nhiệm vụ chính**:
    * **Xử lý Tensor**: Đây là nhiệm vụ chính của Accelerator, thực hiện các phép toán đại số tuyến tính (linear algebra) trên tensor, vốn là cấu trúc dữ liệu cơ bản của các mô hình AI.
    * **MAC Array (Multiply-Accumulate Array)**: Là một mảng lớn các đơn vị tính toán chuyên biệt, thực hiện đồng thời hàng trăm hoặc hàng nghìn phép nhân tích lũy.
    * **Systolic Array**: Một kiến trúc nâng cao của MAC array. Dữ liệu "chảy" qua các đơn vị tính toán một cách tuần tự, giảm thiểu việc phải di chuyển dữ liệu ra ngoài, từ đó tiết kiệm năng lượng và tăng tốc độ.
    * **GRU Cell**: Các đơn vị xử lý phần cứng được thiết kế riêng để xử lý các phép tính phức tạp của một ô GRU, giúp tăng tốc hiệu quả các mô hình ngôn ngữ nhỏ được xây dựng trên kiến trúc này.

### 3.5. Các khối mở rộng theo GPU

Các khối này được tích hợp vào Accelerator để tận dụng tối đa khả năng xử lý song song, một đặc trưng của GPU.

* **Warp/Thread Scheduler**:
    * **Nhiệm vụ**: Quản lý và điều phối hàng nghìn luồng (thread) tính toán nhỏ. Nó sẽ phân chia một phép toán ma trận lớn thành nhiều luồng nhỏ và gán chúng cho các đơn vị tính toán trong Accelerator.
    * **Hiệu quả**: Đảm bảo tất cả các đơn vị tính toán đều hoạt động hiệu quả, tránh tình trạng bị rảnh rỗi.

* **SIMD/SIMT Logic**:
    * **SIMD (Single Instruction, Multiple Data)**: Một lệnh duy nhất có thể xử lý nhiều dữ liệu cùng lúc.
    * **SIMT (Single Instruction, Multiple Threads)**: Một khái niệm nâng cao hơn. Bộ xử lý sẽ điều khiển một nhóm các luồng (thread) để thực hiện cùng một lệnh. 
    * **Nhiệm vụ**: Tận dụng triệt để tính song song của kiến trúc, cho phép các đơn vị tính toán thực hiện cùng một loại phép tính trên nhiều mẩu dữ liệu khác nhau.

* **Multi-stage Pipeline**:
    * **Nhiệm vụ**: Chia quá trình xử lý một lệnh thành nhiều giai đoạn (ví dụ: `Fetch`, `Decode`, `Execute`, `Write Back`).
    * **Hiệu quả**: Trong khi một lệnh đang được thực thi ở giai đoạn `Execute`, lệnh tiếp theo đã bắt đầu ở giai đoạn `Decode`, tạo thành một "dây chuyền sản xuất" liên tục, giúp tăng thông lượng và hiệu suất tổng thể của Accelerator.

## 4. Khối toán học chính
Ba khối toán học sau là những thành phần cốt lõi của các mô hình học sâu hiện đại, đặc biệt là các mô hình ngôn ngữ như SLM.

### 4.1. MAC Array (Multiply-Accumulate Array)

**MAC Array** là một mảng lớn các đơn vị **MAC (Multiply-Accumulate)**, được thiết kế để thực hiện các phép toán nhân tích lũy song song trên ma trận.

* **Phép toán**: Phép toán cơ bản là $a \times b + c$. Nó thực hiện phép nhân của hai số và sau đó cộng kết quả vào một số khác.
* **Mô hình toán học**: Trong các mô hình nơ-ron, đây chính là phép nhân ma trận giữa ma trận đầu vào ($X$) và ma trận trọng số ($W$) của một lớp mạng, sau đó cộng với vector bias ($b$) để tạo ra đầu ra ($Y$). Công thức là $Y = X \times W + b$.
* **Ứng dụng**: Đây là phép toán chiếm phần lớn thời gian tính toán trong các mô hình học sâu, từ mạng truyền thẳng (feedforward networks) cho đến các mạng phức tạp hơn. Việc thực hiện phép toán này trên một mảng song song (array) thay vì tuần tự giúp tăng tốc độ lên hàng trăm, thậm chí hàng nghìn lần.
* **Tối ưu hóa phần cứng**: Accelerator sẽ có các MAC array lớn, có thể được tổ chức dưới dạng **systolic array**. Dữ liệu sẽ "chảy" qua các đơn vị tính toán một cách liên tục, giảm thiểu việc truy cập bộ nhớ, từ đó tiết kiệm năng lượng và tăng hiệu quả. 

### 4.2. GRU/LSTM Unit với LUT Sigmoid/Tanh

Các đơn vị **GRU** và **LSTM** là các thành phần của mạng nơ-ron hồi quy, được thiết kế để xử lý các chuỗi dữ liệu và ghi nhớ thông tin dài hạn.

* **Phép toán**: Các đơn vị này sử dụng nhiều phép nhân ma trận và vector, cùng với các hàm kích hoạt phi tuyến tính như **sigmoid** ($\sigma$) và **tanh**.
    * **GRU**: Công thức bao gồm các phép toán cho cổng cập nhật ($z$) và cổng đặt lại ($r$), cùng với việc tính toán trạng thái ẩn mới ($h_t$).
        * $z_t = \sigma(W_z x_t + U_z h_{t-1})$
        * $r_t = \sigma(W_r x_t + U_r h_{t-1})$
        * $\tilde{h}_t = \tanh(W x_t + U(r_t \odot h_{t-1}))$
        * $h_t = (1-z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$
    * **LSTM**: Cấu trúc tương tự nhưng phức tạp hơn với cổng quên ($f$), cổng đầu vào ($i$), và cổng đầu ra ($o$).
* **Tối ưu hóa phần cứng**:
    * **Tăng tốc phép nhân ma trận**: Các phép toán nhân ma trận-vector có thể được xử lý hiệu quả trên MAC array.
    * **Bảng tra cứu (Lookup Table - LUT)**: Đây là một kỹ thuật tối ưu quan trọng. Thay vì phải tính toán các hàm phi tuyến tính phức tạp như sigmoid và tanh, phần cứng sẽ lưu trữ các giá trị của hàm này trong một bảng tra cứu. Khi cần, nó chỉ cần tra cứu kết quả, giúp tiết kiệm thời gian và tài nguyên tính toán đáng kể so với việc tính toán trực tiếp.

### 4.3. Attention Block (Dot Product + Softmax)

**Cơ chế Attention** là một khối toán học quan trọng trong các mô hình ngôn ngữ dựa trên kiến trúc **Transformer**. Nó cho phép mô hình tập trung vào các phần có ý nghĩa nhất của dữ liệu đầu vào.

* **Phép toán**:
    * **Phép nhân vô hướng (Dot Product)**: Đây là phép toán cốt lõi. Mô hình tính toán "sự tương đồng" giữa các vector truy vấn (query) và các vector khóa (key) bằng cách nhân vô hướng chúng.
        * $Attention(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$
    * **Softmax**: Sau khi tính toán ma trận điểm số (score matrix) từ phép nhân vô hướng, hàm softmax được áp dụng để chuyển các điểm số thành các trọng số có tổng bằng 1, biểu thị mức độ quan trọng của từng vector đầu vào.
* **Tối ưu hóa phần cứng**:
    * **Tăng tốc Dot Product**: Phép toán này cũng là một dạng phép nhân ma trận, có thể được thực hiện hiệu quả trên MAC array hoặc systolic array.
    * **Tối ưu Softmax**: Tương tự như sigmoid/tanh, hàm softmax cũng có thể được tối ưu hóa bằng cách sử dụng bảng tra cứu (LUT) hoặc các mạch logic chuyên biệt để tăng tốc độ tính toán.
    * **Tối ưu băng thông**: Các phép toán Attention đòi hỏi rất nhiều lần truy cập dữ liệu (vector Q, K, V), vì vậy việc tối ưu băng thông bộ nhớ và giảm thiểu việc di chuyển dữ liệu là cực kỳ quan trọng đối với hiệu quả năng lượng và độ trễ.