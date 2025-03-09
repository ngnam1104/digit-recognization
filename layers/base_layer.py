class BaseLayer:
    def forward(self, X):
        raise NotImplementedError
    
    def backward(self, dY, *args):
        """Hàm backward tổng quát, các lớp con sẽ override."""
        raise NotImplementedError