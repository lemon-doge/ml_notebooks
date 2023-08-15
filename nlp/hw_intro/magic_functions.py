def process_frame(line, words):
    vec = [0] * len(words)
    for w in line:
        if w in words:
            vec[words.index(w)] += 1

    return [vec, 1 if len(line) > 30 else 0]
