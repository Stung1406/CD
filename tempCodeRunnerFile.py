train(args):
    """Chế độ train: sinh nhiều puzzle có nghiệm, huấn luyện Q-Learning đa instance,
    rồi lưu Q-table ra file JSON để đánh giá tổng quát hóa ở chế độ eval."