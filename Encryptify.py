import streamlit as st
import string
import numpy as np

# Helper functions for encryption methods
class CipherMethods:

    @staticmethod
    def caesar_encrypt(text, shift):
        result = ""
        for char in text:
            if char.isalpha():
                start = ord('A') if char.isupper() else ord('a')
                result += chr((ord(char) - start + shift) % 26 + start)
            else:
                result += char
        return result

    @staticmethod
    def caesar_decrypt(text, shift):
        return CipherMethods.caesar_encrypt(text, -shift)

    @staticmethod
    def vigenere_encrypt(text, key):
        key = key.lower()
        result = ""
        key_index = 0
        for char in text:
            if char.isalpha():
                start = ord('A') if char.isupper() else ord('a')
                shift = ord(key[key_index % len(key)]) - ord('a')
                result += chr((ord(char) - start + shift) % 26 + start)
                key_index += 1
            else:
                result += char
        return result

    @staticmethod
    def vigenere_decrypt(text, key):
        key = key.lower()
        result = ""
        key_index = 0
        for char in text:
            if char.isalpha():
                start = ord('A') if char.isupper() else ord('a')
                shift = ord(key[key_index % len(key)]) - ord('a')
                result += chr((ord(char) - start - shift) % 26 + start)
                key_index += 1
            else:
                result += char
        return result

    @staticmethod
    def affine_encrypt(text, a, b):
        if np.gcd(a, 26) != 1:
            raise ValueError("Key 'a' must be coprime with 26.")
        result = ""
        for char in text:
            if char.isalpha():
                start = ord('A') if char.isupper() else ord('a')
                result += chr(((a * (ord(char) - start) + b) % 26) + start)
            else:
                result += char
        return result

    @staticmethod
    def affine_decrypt(text, a, b):
        if np.gcd(a, 26) != 1:
            raise ValueError("Key 'a' must be coprime with 26.")
        a_inv = pow(a, -1, 26)
        result = ""
        for char in text:
            if char.isalpha():
                start = ord('A') if char.isupper() else ord('a')
                result += chr(((a_inv * ((ord(char) - start) - b)) % 26) + start)
            else:
                result += char
        return result

# Streamlit App
def main():
    st.title("Cipher Encryption and Decryption")

    cipher_choice = st.selectbox(
        "Select a Cipher Method:",
        ["Caesar Cipher", "Vigenère Cipher", "Affine Cipher"]
    )

    st.subheader(f"{cipher_choice}")

    text = st.text_area("Enter Text:", "", help="Input the text you want to encrypt or decrypt.")

    if cipher_choice == "Caesar Cipher":
        shift = st.number_input("Enter Shift Key:", min_value=1, max_value=25, value=3)
        mode = st.radio("Select Mode:", ["Encrypt", "Decrypt"], horizontal=True)

        if st.button("Run Caesar Cipher"):
            if mode == "Encrypt":
                result = CipherMethods.caesar_encrypt(text, shift)
                st.success(f"Encrypted Text: {result}")
            else:
                result = CipherMethods.caesar_decrypt(text, shift)
                st.success(f"Decrypted Text: {result}")

    elif cipher_choice == "Vigenère Cipher":
        key = st.text_input("Enter Keyword:", help="Input a keyword for encryption/decryption.").strip()
        mode = st.radio("Select Mode:", ["Encrypt", "Decrypt"], horizontal=True)

        if st.button("Run Vigenère Cipher"):
            try:
                if mode == "Encrypt":
                    result = CipherMethods.vigenere_encrypt(text, key)
                    st.success(f"Encrypted Text: {result}")
                else:
                    result = CipherMethods.vigenere_decrypt(text, key)
                    st.success(f"Decrypted Text: {result}")
            except Exception as e:
                st.error(f"Error: {e}")

    elif cipher_choice == "Affine Cipher":
        a = st.number_input("Enter 'a' (must be coprime with 26):", min_value=1, value=5)
        b = st.number_input("Enter 'b' (integer key):", value=8)
        mode = st.radio("Select Mode:", ["Encrypt", "Decrypt"], horizontal=True)

        if st.button("Run Affine Cipher"):
            try:
                if mode == "Encrypt":
                    result = CipherMethods.affine_encrypt(text, a, b)
                    st.success(f"Encrypted Text: {result}")
                else:
                    result = CipherMethods.affine_decrypt(text, a, b)
                    st.success(f"Decrypted Text: {result}")
            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
