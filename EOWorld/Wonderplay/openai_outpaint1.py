import argparse
import base64
import os
import tempfile
import urllib.request
from pathlib import Path

from PIL import Image, ImageChops, ImageStat


DEFAULT_IMAGE = (
    "3d_result/wonderplay/venice/Gen-02-09_21-09-32/"
    "multiview_000/generation_condition.png"
)
DEFAULT_PROMPT = (
    "Outpaint the missing area as a seamless continuation of the existing scene. "
    "Match the perspective, composition, lighting, color tone, texture, and visual "
    "style of the original image. Preserve the visible content and integrate the "
    "completed region naturally, with coherent depth, realistic details, and smooth "
    "transitions."
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="OpenAI GPT Image outpaint experiment matching outpaint1.py."
    )
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Input image path.")
    parser.add_argument(
        "--mask",
        default=None,
        help=(
            "Optional white-edit/black-keep mask. If omitted, the script first uses "
            "generation_hole_mask.png next to the input image, then falls back to "
            "detecting near-white blank pixels."
        ),
    )
    parser.add_argument("--output", default="outpaint_openai.png", help="Output path.")
    parser.add_argument("--model", default="gpt-image-2", help="OpenAI image model.")
    parser.add_argument("--size", default="1024x1024", help="Requested API output size.")
    parser.add_argument("--quality", default="medium", help="low, medium, high, or auto.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Edit prompt.")
    parser.add_argument(
        "--negative-prompt",
        default="",
        help="OpenAI Images API has no SD-style negative_prompt; this is appended as Avoid: ...",
    )
    parser.add_argument(
        "--white-threshold",
        type=int,
        default=250,
        help="Threshold for deriving a mask from white blank pixels.",
    )
    parser.add_argument(
        "--resize-output-to-input",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resize API output back to the input image size before saving.",
    )
    parser.add_argument(
        "--invert-mask-alpha",
        action="store_true",
        help="Invert the mask alpha if your API/model expects transparent edit regions.",
    )
    parser.add_argument(
        "--save-debug-mask",
        default="openai_outpaint_mask_alpha.png",
        help="Where to save the RGBA mask sent to the API. Use empty string to skip.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and save the API mask, but do not call OpenAI.",
    )
    return parser.parse_args()


def load_image(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input image not found: {path}")
    return Image.open(path).convert("RGB"), path


def derive_white_mask(image, threshold):
    channels = image.split()
    masks = [channel.point(lambda value: 255 if value >= threshold else 0) for channel in channels]
    mask = ImageChops.multiply(ImageChops.multiply(masks[0], masks[1]), masks[2])
    return mask.convert("L")


def load_or_create_mask(image, image_path, mask_path, threshold):
    if mask_path:
        path = Path(mask_path)
        if not path.is_file():
            raise FileNotFoundError(f"Mask image not found: {path}")
        mask = Image.open(path).convert("L")
        source = path
    else:
        sibling = image_path.with_name("generation_hole_mask.png")
        if sibling.is_file():
            mask = Image.open(sibling).convert("L")
            source = sibling
        else:
            mask = derive_white_mask(image, threshold)
            source = f"derived from white pixels >= {threshold}"

    if mask.size != image.size:
        resampling = getattr(Image, "Resampling", Image)
        mask = mask.resize(image.size, resampling.NEAREST)
    return mask, source


def mask_to_api_rgba(mask, invert_alpha=False):
    alpha = ImageChops.invert(mask) if invert_alpha else mask
    rgba = mask.convert("RGBA")
    rgba.putalpha(alpha)
    return rgba


def describe_mask(mask):
    stat = ImageStat.Stat(mask)
    hist = mask.histogram()
    white = sum(hist[250:])
    nonzero = sum(hist[1:])
    total = mask.size[0] * mask.size[1]
    return {
        "mean": round(stat.mean[0], 3),
        "nonzero": nonzero,
        "white": white,
        "total": total,
        "coverage": round(nonzero / total, 4) if total else 0,
        "bbox": mask.getbbox(),
    }


def build_prompt(prompt, negative_prompt):
    negative_prompt = negative_prompt.strip()
    if not negative_prompt:
        return prompt
    return f"{prompt}\n\nAvoid: {negative_prompt}"


def decode_image_response(result_b64):
    from io import BytesIO

    return Image.open(BytesIO(base64.b64decode(result_b64))).convert("RGB")


def load_output_image(image_data):
    image_base64 = getattr(image_data, "b64_json", None)
    if image_base64:
        return decode_image_response(image_base64)

    image_url = getattr(image_data, "url", None)
    if image_url:
        from io import BytesIO

        with urllib.request.urlopen(image_url) as response:
            return Image.open(BytesIO(response.read())).convert("RGB")

    raise RuntimeError("OpenAI response did not include b64_json or url image data.")


def main():
    args = parse_args()
    if not args.dry_run and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    image, image_path = load_image(args.image)
    mask, mask_source = load_or_create_mask(
        image=image,
        image_path=image_path,
        mask_path=args.mask,
        threshold=args.white_threshold,
    )
    mask_rgba = mask_to_api_rgba(mask, invert_alpha=args.invert_mask_alpha)
    prompt = build_prompt(args.prompt, args.negative_prompt)

    print(f"image: {image_path} {image.mode} {image.size}")
    print(f"mask: {mask_source} {mask.mode} {mask.size} {describe_mask(mask)}")
    print(f"model: {args.model}, size: {args.size}, quality: {args.quality}")
    print(f"output: {Path(args.output).resolve()}")

    if args.save_debug_mask:
        debug_mask_path = Path(args.save_debug_mask)
        mask_rgba.save(debug_mask_path, format="PNG")
        print(f"debug mask saved at {debug_mask_path.resolve()}")

    if args.dry_run:
        print("dry run complete; OpenAI API was not called.")
        return

    from openai import OpenAI

    client = OpenAI()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        api_image_path = tmpdir / "input.png"
        api_mask_path = tmpdir / "mask.png"
        image.save(api_image_path, format="PNG")
        mask_rgba.save(api_mask_path, format="PNG")

        with open(api_image_path, "rb") as image_file, open(api_mask_path, "rb") as mask_file:
            result = client.images.edit(
                model=args.model,
                image=image_file,
                mask=mask_file,
                prompt=prompt,
                size=args.size,
                quality=args.quality,
                output_format="png",
            )

    output_image = load_output_image(result.data[0])
    if args.resize_output_to_input and output_image.size != image.size:
        resampling = getattr(Image, "Resampling", Image)
        output_image = output_image.resize(image.size, resampling.LANCZOS)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_image.save(output_path)
    print(f"image saved at {output_path.resolve()}")


if __name__ == "__main__":
    main()
