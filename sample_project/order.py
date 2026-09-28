def create_order(user_id, products):
    total = calculate_total(products)
    return save_order(user_id, products, total)