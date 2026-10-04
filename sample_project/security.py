import hashlib


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(password, stored_password_hash):
    generated_hash = hash_password(password)
    return generated_hash == stored_password_hash