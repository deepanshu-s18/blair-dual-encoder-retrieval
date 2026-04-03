"""Dataset preparation script."""
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=10000)
    args = parser.parse_args()
    print(f"Preparing dataset with size {args.size}")

if __name__ == "__main__":
    main()

def clean_text(text: str) -> str:
    import re
    text = text.lower().strip()
    return re.sub(r'\s+', ' ', text)
