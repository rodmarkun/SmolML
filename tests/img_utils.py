import os

# All test-generated images go into test-imgs/ at the repository root
TEST_IMGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test-imgs")

def image_path(filename):
    """Return the full path for a test-generated image inside test-imgs/, creating the folder if needed."""
    os.makedirs(TEST_IMGS_DIR, exist_ok=True)
    return os.path.join(TEST_IMGS_DIR, filename)
