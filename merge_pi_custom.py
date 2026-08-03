import csv
import json
import sys
import uuid

from recover_utc import recover_utc_timestamp

OPERATOR_ID = "intern"
STATION_ID = "household"
ROBOT_ID = "human"

FIELDNAMES = [
    "episode_index",
    "operator_id",
    "is_eval_episode",
    "episode_id",
    "start_timestamp",
    "checkpoint_path",
    "success",
    "station_id",
    "robot_id",
]


def load_episodes(jsonl_path):
    episodes = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            episodes.append(json.loads(line))
    return episodes


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <episodes.jsonl> [output.csv]")
        sys.exit(1)

    jsonl_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "custom_metadata.csv"

    episodes = load_episodes(jsonl_path)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(FIELDNAMES)

        for episode in episodes:
            entry = []

            # 1. episode index
            entry.append(episode["episode_index"])

            # 2. operator id
            entry.append(OPERATOR_ID)

            # 3. is_eval_episode
            entry.append(False)

            # 4. episode_id
            entry.append(str(uuid.uuid4()))

            # 5. start_timestamp    float   UTC seconds (Unix epoch time)
            entry.append(recover_utc_timestamp(episode["tasks"][0], episode["length"]))

            # 6. checkpoint_path (only for eval episodes)
            entry.append(None)

            # 7. success    boolean   Whether episode was successful
            entry.append(True)

            # 8. station_id   string   Station/scene identifier
            entry.append(STATION_ID)

            # 9. robot_id
            entry.append(ROBOT_ID)

            writer.writerow(entry)

    print(f"Wrote {len(episodes)} rows to {output_path}")


if __name__ == "__main__":
    main()