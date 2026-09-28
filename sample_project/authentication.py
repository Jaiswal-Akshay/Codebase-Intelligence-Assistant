def login(email, password):
    user = find_user_by_email(email)

    if user and verify_password(password, user["password_hash"]):
        return create_session(user)

    return None