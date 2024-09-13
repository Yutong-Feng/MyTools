import argparse
import json
import os

from pandas import DataFrame, concat


class ConfigFactory:
    META_PATH = "configs"
    DEFAULT_FILE = "default.json"
    # DEFAULT_FILE = "baseline.json"

    @staticmethod
    def load_json(file_path):
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                return json.load(file)
        else:
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    @staticmethod
    def build_args_from_config(config):
        parser = argparse.ArgumentParser()
        for key, value in config.items():
            value_type = type(value).__name__
            if isinstance(value_type, (bool, int, float, str)):
                parser.add_argument(f"--{key}",
                                    type=type(value),
                                    default=value)
            else:
                raise TypeError(
                    f"Unsupported type '{value_type}' for key '{key}'")
        return parser.parse_args()

    @staticmethod
    def build_message_df(cfg):
        msg_df = DataFrame.from_dict(cfg.items(), orient="columns")
        msg_df.columns = ["arg_name", "value"]
        return msg_df

    @classmethod
    def build(cls):
        # Priority: cmd_input > model.json > default.json
        # terminal_parser = argparse.ArgumentParser()
        # terminal_parser.add_argument("--config_file", type=str, default="")
        # terminal_args = terminal_parser.parse_args()

        default_config = cls.load_json(
            os.path.join(cls.META_PATH, cls.DEFAULT_FILE))

        # if terminal_args.config_file == "":
        #     config_file = default_config["config_file"] + ".json"
        # else:
        #     config_file = terminal_args.config_file + ".json"
        config_file = input("Input config file: ")
        if config_file == "":
            config_file = "PQFormer.json"
        else:
            config_file = config_file + ".json"
        print(f"Get config file from {config_file}")

        new_config = cls.load_json(os.path.join(cls.META_PATH, config_file))
        using_config = {**default_config, **new_config}
        args = cls.build_args_from_config(using_config)

        # Print args message
        message_df = (cls.build_message_df(
            vars(args)).sort_values(by="arg_name").reset_index(drop=True))
        message_df["source"] = "terminal"

        default_df = cls.build_message_df(default_config)
        default_df["source"] = cls.DEFAULT_FILE
        new_df = cls.build_message_df(new_config)
        new_df["source"] = config_file

        json_df = (concat(
            [
                new_df,
                default_df[~default_df["arg_name"].isin(new_df["arg_name"])]
            ],
            ignore_index=True,
        ).sort_values(by="arg_name").reset_index(drop=True))
        message_df["source"] = json_df["source"].where(
            json_df["value"] == message_df["value"], message_df["source"])
        message_df["state"] = "USED"

        message = "\n" + message_df.to_string(index=False)
        return (
            argparse.Namespace(**message_df[message_df["state"] != "FORBIDDEN"]
                               [["arg_name", "value"]].set_index(
                                   "arg_name").to_dict()["value"]),
            message,
        )


if __name__ == "__main__":
    args_product, msg = ConfigFactory.build()
    print(msg)
    print(args_product)
    print("Finish!")
