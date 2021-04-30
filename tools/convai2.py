from laser.data.utils import PERSONA_SEP_CHAR


def convert(inp_file: str, src_file: str, tgt_file: str):
    source, target = [], []
    your_personas, partner_personas = [], []
    with open(inp_file, "r") as f:
        for line in f:
            idx, sent = line.rstrip("\n").split(" ", maxsplit=1)
            if int(idx) == 1:
                your_personas, partner_personas = [], []

            if sent.startswith("your persona: "):
                your_personas.append(sent[len("your persona: ") :])
            elif sent.startswith("partner's persona: "):
                partner_personas.append(sent[len("partner's persona: ") :])
            else:
                x, y = sent.split("\t")
                # Add your_personas to source, process the data as translation task
                # Using this PERSONA_SEP_CHAR to split out the personas in model
                source.append(PERSONA_SEP_CHAR.join([x] + your_personas))
                target.append(y)

    def write_file(sents, oup_file):
        with open(oup_file, "w") as f:
            for line in sents:
                f.write(line)
                f.write("\n")

    write_file(source, src_file)
    write_file(target, tgt_file)


if __name__ == "__main__":
    convert(
        "./data/convai2/train_both_original_no_cands.txt",
        "./data/convai2/raw/en1-en2.en1",
        "./data/convai2/raw/en1-en2.en2",
    )
