#!/usr/bin/env python
__author__ = "Enzo Ubaldo Petrocco"
import sys


sys.path.insert(1, "../../")

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from Utils.Results.Results_only_ADV02 import ResAcquisitionClass
from Utils.FileManager.FileManager import FileManagerClass
import functools
import matplotlib.colors as mcolors

taus = [0.1, 0.3, 0.5]
basePath = "../Results/temps2/"


class VisualizerClass:
    def plot_table(self, df, title):
        """
        This function plots a DataFrame as Table
        :param df: DataFrame
        :param title: title of Table
        """
        fig, ax = plt.subplots()
        # hide axes
        fig.patch.set_visible(False)
        ax.axis("off")
        ax.axis("tight")
        tab = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
        tab.auto_set_font_size(False)
        tab.set_fontsize(9.5)
        fig.tight_layout()
        ax.set_title(title)
        plt.show()


def print_tables(
    standards,
    lamps,
    cultures,
    percents,
    augments,
    adversary,
    lambda_indeces,
    taugments,
    tadversaries,
    test_g_augs,
    test_eps,
    paths,
    visobj,
):
    for standard in standards:
        for lamp in lamps:
            for culture in cultures:
                for percent in range(len(percents)):
                    for augment in augments:
                        for adv in adversary:
                            if standard:
                                for taugment in taugments:
                                    for tadversary in tadversaries:
                                        if taugment and tadversary:
                                            for tgaug in range(len(test_g_augs)):
                                                for teps in range(len(test_eps)):

                                                    pt = paths[standard][lamp][culture][
                                                        percent
                                                    ][augment][adv][taugment][
                                                        tadversary
                                                    ][
                                                        tgaug
                                                    ][
                                                        teps
                                                    ]

                                                    pt = pt + "res.csv"
                                                    df = pd.read_csv(pt)
                                                    visobj.plot_table(
                                                        df[
                                                            df.columns[
                                                                1 : len(df.columns)
                                                            ]
                                                        ],
                                                        pt,
                                                    )
                                        if taugment and not tadversary:
                                            for tgaug in range(len(test_g_augs)):
                                                pt = paths[standard][lamp][culture][
                                                    percent
                                                ][augment][adv][taugment][tadversary][
                                                    tgaug
                                                ][
                                                    teps
                                                ]
                                                pt = pt + "res.csv"
                                                df = pd.read_csv(pt)
                                                visobj.plot_table(
                                                    df[df.columns[1 : len(df.columns)]],
                                                    pt,
                                                )

                                        if not taugment and tadversary:
                                            for teps in range(len(test_eps)):
                                                pt = paths[standard][lamp][culture][
                                                    percent
                                                ][augment][adv][taugment][tadversary][
                                                    tgaug
                                                ][
                                                    teps
                                                ]
                                                pt = pt + "res.csv"
                                                df = pd.read_csv(pt)
                                                visobj.plot_table(
                                                    df[df.columns[1 : len(df.columns)]],
                                                    pt,
                                                )
                                        if not taugment and not tadversary:
                                            pt = paths[standard][lamp][culture][
                                                percent
                                            ][augment][adv][taugment][tadversary][
                                                tgaug
                                            ][
                                                teps
                                            ]
                                            pt = pt + "res.csv"
                                            df = pd.read_csv(pt)
                                            visobj.plot_table(
                                                df[df.columns[1 : len(df.columns)]],
                                                pt,
                                            )
                            else:
                                for lambda_index in lambda_indeces:
                                    for taugment in taugments:
                                        for tadversary in tadversaries:
                                            if taugment and tadversary:
                                                for tgaug in range(len(test_g_augs)):
                                                    for teps in range(len(test_eps)):
                                                        pt = paths[standard][lamp][
                                                            culture
                                                        ][percent][augment][adv][
                                                            lambda_index
                                                        ][
                                                            taugment
                                                        ][
                                                            tadversary
                                                        ][
                                                            tgaug
                                                        ][
                                                            teps
                                                        ]

                                                        pt = pt + "res.csv"
                                                        df = pd.read_csv(pt)
                                                        visobj.plot_table(
                                                            df[
                                                                df.columns[
                                                                    1 : len(df.columns)
                                                                ]
                                                            ],
                                                            pt,
                                                        )
                                            if taugment and not tadversary:
                                                for tgaug in range(len(test_g_augs)):
                                                    pt = paths[standard][lamp][culture][
                                                        percent
                                                    ][augment][adv][lambda_index][
                                                        taugment
                                                    ][
                                                        tadversary
                                                    ][
                                                        tgaug
                                                    ][
                                                        teps
                                                    ]
                                                    pt = pt + "res.csv"
                                                    df = pd.read_csv(pt)
                                                    visobj.plot_table(
                                                        df[
                                                            df.columns[
                                                                1 : len(df.columns)
                                                            ]
                                                        ],
                                                        pt,
                                                    )
                                            if not taugment and tadversary:
                                                for teps in range(len(test_eps)):
                                                    pt = paths[standard][lamp][culture][
                                                        percent
                                                    ][augment][adv][lambda_index][
                                                        taugment
                                                    ][
                                                        tadversary
                                                    ][
                                                        tgaug
                                                    ][
                                                        teps
                                                    ]
                                                    pt = pt + "res.csv"
                                                    df = pd.read_csv(pt)
                                                    visobj.plot_table(
                                                        df[
                                                            df.columns[
                                                                1 : len(df.columns)
                                                            ]
                                                        ],
                                                        pt,
                                                    )
                                            if not taugment and not tadversary:
                                                tgaug = 0
                                                teps = 0
                                                pt = paths[standard][lamp][culture][
                                                    percent
                                                ][augment][adv][lambda_index][taugment][
                                                    tadversary
                                                ][
                                                    tgaug
                                                ][
                                                    teps
                                                ]
                                                pt = pt + "res.csv"
                                                df = pd.read_csv(pt)
                                                visobj.plot_table(
                                                    df[df.columns[1 : len(df.columns)]],
                                                    pt,
                                                )


def perstr2float(val):
    val = val.split("%")
    val = val[0]
    val = float(val)
    return val


def plotandsave(
    stddf, mitdfs, plot=True, title="", save=False, path="./", weighted=True
):
    """
    This function plots in CIC-ERR plane the CIC(ERR) value of standard model
    and the CIC(ERR, gamma) values of mitigated model
    :param stddf: DataFrame of standard model
    :param mitdfs: DataFrame of mitigated model
    :param save: if enable, saves the image in path
    :param path: path in which the image is saved
    """
    ERRString = "ERR"
    if weighted:
        ERRString = "W_ERR"
    X = []
    y = []
    X_err = []
    y_err = []
    for i in range(len(mitdfs)):
        X.append(perstr2float(mitdfs[i][f"{ERRString}"][0]))
        y.append(perstr2float(mitdfs[i]["CIC"][0]))
        X_err.append(perstr2float(mitdfs[i][f"{ERRString} std"][0]))
        y_err.append(perstr2float(mitdfs[i]["CIC std"][0]))

    # Plotting both the curves simultaneously
    plt.scatter(
        perstr2float(stddf[f"{ERRString}"][0]),
        perstr2float(stddf["CIC"][0]),
        color="r",
        label="Standard Model",
    )
    plt.scatter(X, y, color="k", label="Mitigated Model")

    # Plot errors
    plt.errorbar(
        perstr2float(stddf[f"{ERRString}"][0]),
        perstr2float(stddf["CIC"][0]),
        xerr=perstr2float(stddf[f"{ERRString} std"][0]),
        yerr=perstr2float(stddf["CIC std"][0]),
        fmt="ro",
    )
    plt.errorbar(X, y, xerr=X_err, yerr=y_err, fmt="ko")

    # Naming the x-axis, y-axis and the whole graph
    plt.xlabel(f"{ERRString}")
    plt.ylabel("CIC")
    plt.title(title)

    plt.xlim((0, 100))
    plt.ylim((0, 30))

    # Adding legend, which helps us recognize the curve according to it's color
    plt.legend()

    # To load the display window
    if plot:
        plt.show()

    if save:
        print(f"Picture saved in {path}")
        plt.savefig(path)

    # print(path)
    plt.close()


def gen_title_plot(
    lamp, culture, percent, augment, adversary, taugment, tadversary, gaug, geps
):
    test_g_augs = [0.01, 0.05, 0.1]
    test_eps = [0.0005, 0.001, 0.005]
    svpath = "./"
    if lamp:
        if culture == 0:
            title = "Chinese Model"
            svpath = "LC/"
        if culture == 1:
            title = "French Model"
            svpath = "LF/"
        if culture == 2:
            title = "Turkish Model"
            svpath = "LT/"
    else:
        if culture == 0:
            title = "Indian Model"
            svpath = "CI/"
        if culture == 1:
            title = "Japanese Model"
            svpath = "CJ/"
        if culture == 2:
            title = "Scandinavian Model"
            svpath = "CS/"

    title += " with μ" + f"={percent}\n"
    svpath += f"{percent}/"
    if augment:
        if adversary:
            title += "Using TOTAUG in training "
            svpath += "TOTAUG/"
        else:
            title += "Using STDAUG in training "
            svpath += "STDAUG/"
    else:

        if adversary:
            title += "Using ADVAUG in training "
            svpath += "ADVAUG/"
        else:
            title += "Using NOAUG in training "
            svpath += "NOAUG/"

    if taugment:
        if tadversary:
            title += f"Tested on TOTAUG with GAUG={gaug} and ε={geps}"
            svpath += f"TTOTAUG/GAUG={gaug}/EPS={geps}/"
        else:
            title += f"Tested on STDAUG with GAUG={gaug}"
            svpath += f"TSTDAUG/GAUG={gaug}/"
    else:
        if tadversary:
            title += f"Tested on ADVAUG with ε={geps}"
            svpath += f"TADVAUG/EPS={geps}/"
        else:
            title += f"Tested on NOAUG"
            svpath += f"TNOAUG/"

    fObj = FileManagerClass(svpath)
    del fObj
    svpath += "img.png"
    # (f"svpath is {svpath}")

    return title, svpath


def graphic_lambdas_comparison(
    standards,
    lamps,
    cultures,
    percents,
    augments,
    adversary,
    lambda_indeces,
    taugments,
    tadversaries,
    test_g_augs,
    test_eps,
    resacqobj: ResAcquisitionClass,
    basePath="../Results/",
    plot=False,
    imb=0,
):
    for lamp in lamps:
        for culture in cultures:
            for percent in percents:
                for augment in augments:
                    for adv in adversary:
                        for taugment in taugments:
                            for tadversary in tadversaries:
                                if taugment and tadversary:
                                    for tgaug in test_g_augs:
                                        for teps in test_eps:
                                            pt = resacqobj.buildPath(
                                                basePath=basePath,
                                                standard=1,
                                                alg="DL",
                                                lamp=lamp,
                                                culture=culture,
                                                percent=percent,
                                                augment=augment,
                                                adversary=adv,
                                                lambda_index=0,
                                                taugment=taugment,
                                                tadversary=tadversary,
                                                tgaug=tgaug,
                                                teps=teps,
                                                imbalanced=imb,
                                            )

                                            pt = pt.split("/")
                                            pt = pt[0 : len(pt) - 2]
                                            stdpt = ""
                                            for p in pt:
                                                stdpt += p + "/"
                                            stdpt += "res.csv"
                                            stddf = pd.read_csv(stdpt)
                                            mitdfs = []
                                            for lambda_index in lambda_indeces:
                                                pt = resacqobj.buildPath(
                                                    basePath=basePath,
                                                    standard=0,
                                                    alg="DL",
                                                    lamp=lamp,
                                                    culture=culture,
                                                    percent=percent,
                                                    augment=augment,
                                                    adversary=adv,
                                                    lambda_index=lambda_index,
                                                    taugment=taugment,
                                                    tadversary=tadversary,
                                                    tgaug=tgaug,
                                                    teps=teps,
                                                    imbalanced=imb,
                                                )
                                                pt = pt.split("/")
                                                pt = pt[: len(pt) - 2]
                                                mitpt = ""
                                                for p in pt:
                                                    mitpt += p + "/"
                                                pt = mitpt + "res.csv"
                                                df = pd.read_csv(pt)
                                                mitdfs.append(df)

                                            title, svpath = gen_title_plot(
                                                lamp,
                                                culture,
                                                percent,
                                                augment,
                                                adv,
                                                taugment,
                                                tadversary,
                                                tgaug,
                                                teps,
                                            )
                                            plotandsave(
                                                stddf=stddf,
                                                mitdfs=mitdfs,
                                                plot=plot,
                                                title=title,
                                                save=True,
                                                path=svpath,
                                            )

                                if taugment and not tadversary:
                                    for tgaug in test_g_augs:
                                        pt = resacqobj.buildPath(
                                            basePath=basePath,
                                            standard=1,
                                            alg="DL",
                                            lamp=lamp,
                                            culture=culture,
                                            percent=percent,
                                            augment=augment,
                                            adversary=adv,
                                            lambda_index=0,
                                            taugment=taugment,
                                            tadversary=tadversary,
                                            tgaug=tgaug,
                                            teps=0,
                                            imbalanced=imb,
                                        )

                                        pt = pt.split("/")
                                        pt = pt[0 : len(pt) - 2]
                                        stdpt = ""
                                        for p in pt:
                                            stdpt += p + "/"
                                        stdpt += "res.csv"
                                        stddf = pd.read_csv(stdpt)
                                        mitdfs = []
                                        for lambda_index in lambda_indeces:
                                            pt = resacqobj.buildPath(
                                                basePath=basePath,
                                                standard=0,
                                                alg="DL",
                                                lamp=lamp,
                                                culture=culture,
                                                percent=percent,
                                                augment=augment,
                                                adversary=adv,
                                                lambda_index=lambda_index,
                                                taugment=taugment,
                                                tadversary=tadversary,
                                                tgaug=tgaug,
                                                teps=0,
                                                imbalanced=imb,
                                            )
                                            pt = pt.split("/")
                                            pt = pt[: len(pt) - 2]
                                            mitpt = ""
                                            for p in pt:
                                                mitpt += p + "/"
                                            pt = mitpt + "res.csv"
                                            df = pd.read_csv(pt)
                                            mitdfs.append(df)

                                        title, svpath = gen_title_plot(
                                            lamp,
                                            culture,
                                            percent,
                                            augment,
                                            adv,
                                            taugment,
                                            tadversary,
                                            tgaug,
                                            0,
                                        )
                                        plotandsave(
                                            stddf=stddf,
                                            mitdfs=mitdfs,
                                            plot=plot,
                                            title=title,
                                            save=True,
                                            path=svpath,
                                        )

                                if not taugment and tadversary:
                                    for teps in test_eps:
                                        pt = resacqobj.buildPath(
                                            basePath=basePath,
                                            standard=1,
                                            alg="DL",
                                            lamp=lamp,
                                            culture=culture,
                                            percent=percent,
                                            augment=augment,
                                            adversary=adv,
                                            lambda_index=0,
                                            taugment=taugment,
                                            tadversary=tadversary,
                                            tgaug=0,
                                            teps=teps,
                                            imbalanced=imb,
                                        )
                                        pt = pt.split("/")
                                        pt = pt[0 : len(pt) - 2]
                                        stdpt = ""
                                        for p in pt:
                                            stdpt += p + "/"
                                        stdpt += "res.csv"
                                        stddf = pd.read_csv(stdpt)
                                        mitdfs = []
                                        for lambda_index in lambda_indeces:
                                            pt = resacqobj.buildPath(
                                                basePath=basePath,
                                                standard=0,
                                                alg="DL",
                                                lamp=lamp,
                                                culture=culture,
                                                percent=percent,
                                                augment=augment,
                                                adversary=adv,
                                                lambda_index=lambda_index,
                                                taugment=taugment,
                                                tadversary=tadversary,
                                                tgaug=0,
                                                teps=teps,
                                                imbalanced=imb,
                                            )
                                            pt = pt.split("/")
                                            pt = pt[: len(pt) - 2]
                                            mitpt = ""
                                            for p in pt:
                                                mitpt += p + "/"
                                            pt = mitpt + "res.csv"
                                            df = pd.read_csv(pt)
                                            mitdfs.append(df)

                                        title, svpath = gen_title_plot(
                                            lamp,
                                            culture,
                                            percent,
                                            augment,
                                            adv,
                                            taugment,
                                            tadversary,
                                            0,
                                            teps,
                                        )
                                        plotandsave(
                                            stddf=stddf,
                                            mitdfs=mitdfs,
                                            plot=plot,
                                            title=title,
                                            save=True,
                                            path=svpath,
                                        )

                                if (not taugment) and (not tadversary):
                                    pt = resacqobj.buildPath(
                                        basePath=basePath,
                                        standard=1,
                                        alg="DL",
                                        lamp=lamp,
                                        culture=culture,
                                        percent=percent,
                                        augment=augment,
                                        adversary=adv,
                                        lambda_index=0,
                                        taugment=taugment,
                                        tadversary=tadversary,
                                        tgaug=0,
                                        teps=0,
                                        imbalanced=imb,
                                    )
                                    pt = pt.split("/")
                                    pt = pt[0 : len(pt) - 2]
                                    stdpt = ""
                                    for p in pt:
                                        stdpt += p + "/"
                                    stdpt += "res.csv"
                                    stddf = pd.read_csv(stdpt)
                                    mitdfs = []
                                    for lambda_index in lambda_indeces:
                                        pt = resacqobj.buildPath(
                                            basePath=basePath,
                                            standard=0,
                                            alg="DL",
                                            lamp=lamp,
                                            culture=culture,
                                            percent=percent,
                                            augment=augment,
                                            adversary=adv,
                                            lambda_index=lambda_index,
                                            taugment=taugment,
                                            tadversary=tadversary,
                                            tgaug=0,
                                            teps=0,
                                            imbalanced=imb,
                                        )
                                        pt = pt.split("/")
                                        pt = pt[: len(pt) - 2]
                                        mitpt = ""
                                        for p in pt:
                                            mitpt += p + "/"
                                        pt = mitpt + "res.csv"
                                        df = pd.read_csv(pt)
                                        mitdfs.append(df)
                                    title, svpath = gen_title_plot(
                                        lamp,
                                        culture,
                                        percent,
                                        augment,
                                        adv,
                                        taugment,
                                        tadversary,
                                        0,
                                        0,
                                    )
                                    plotandsave(
                                        stddf=stddf,
                                        mitdfs=mitdfs,
                                        plot=plot,
                                        title=title,
                                        save=True,
                                        path=svpath,
                                    )


def sort_errs(errs):
    res = []
    errs = np.asarray(errs, dtype=object)

    for i in range(len(errs)):
        err = min(errs[:, 0])
        j = np.where(errs[:, 0] == err)[0][0]
        idx = errs[j, 1]
        res.append([err, idx])
        errs = np.delete(errs, j, 0)

    res = np.asarray(res)
    return res


def get_best_df_gamma(mitdfs, tau=0.3, weighted=True):
    ERRString = "ERR"
    if weighted:
        ERRString = "W_ERR"
    errs = []
    cics = []
    n = int(len(mitdfs) * tau)
    for i, mitdf in enumerate(mitdfs):
        err = perstr2float(mitdf[f"{ERRString}"][0])
        if err > 0:
            errs.append([err, i])
            cic = perstr2float(mitdf["CIC"][0])
            cics.append(cic)
        else:
            errs.append([101, i])
            cics.append(100)

    errs = np.asarray(errs, dtype=object)

    tempcics = [cics[int(i)] for i in errs[:, 1]]

    cicstar = min(tempcics)
    idx = cics.index(cicstar)
    return mitdfs[idx], idx


def get_test_name(taug, tadv, gaug=0, geps=0):
    if taug:
        if tadv:
            name = f"AUG={gaug}, ADV={geps}"
        else:
            name = f"AUG={gaug}"
    else:
        if tadv:
            name = f"ADV={geps}"
        else:
            name = f"NOAUG"

    return name


class Res2TabClass:
    def borderTop(self, strRow, srRow):
        # print(strRow, type(sRow), sRow.name)
        """if srRow.index%4 == 0:
            return [ "border-top: 1pt solid black; font-weight: bold" for sCol in srRow ]
        else:
            return [""] * len(srRow)"""
        return ["border-top: 1pt solid black; font-weight: bold" for sCol in srRow]

    def convert2list(self, test_name, df):
        res = [
            test_name,
            perstr2float(df["ERR^CULTURE 0"][0]),
            perstr2float(df["ERR^CULTURE 0 std"][0]),
            perstr2float(df["ERR^CULTURE 1"][0]),
            perstr2float(df["ERR^CULTURE 1 std"][0]),
            perstr2float(df["ERR^CULTURE 2"][0]),
            perstr2float(df["ERR^CULTURE 2 std"][0]),
            perstr2float(df["ERR"][0]),
            perstr2float(df["ERR std"][0]),
            perstr2float(df["W_ERR"][0]),
            perstr2float(df["W_ERR std"][0]),
            perstr2float(df["CIC"][0]),
            perstr2float(df["CIC std"][0]),
        ]
        return res

    def get_path(self, base, lamp, culture, percent, imbalanced):
        pt = base
        if imbalanced:
            pt += "/IMB/"
        else:
            pt += "/BAL/"
        if lamp:
            if culture == 0:
                pt += "LC/"
            if culture == 1:
                pt += "LF/"
            if culture == 2:
                pt += "LT/"
        else:
            if culture == 0:
                pt += "CI/"
            if culture == 1:
                pt += "CJ/"
            if culture == 2:
                pt += "CS/"

        pt += str(percent) + "/"

        return pt

    def visualize(
        self,
        df,
        title="BLA",
        plot=True,
        save=False,
        path="./",
        element_percentage={"g_augs": 0, "eps": 0},
    ):
        j = 0
        names = sorted(
            mcolors.CSS4_COLORS,
            key=lambda c: tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(c))),
        )
        TSs = df["TS"]
        errs = df["ERR"]
        cics = df["CIC"]

        NOAUG_X = errs[0]
        NOAUG_Y = cics[0]
        plt.scatter(NOAUG_X, NOAUG_Y, color=f"{names[j]}", label="NOAUG")
        j = j + 4

        AUG_X = errs[1 : element_percentage["g_augs"] + 1]
        AUG_Y = cics[1 : element_percentage["g_augs"] + 1]
        #plt.plot(AUG_X, AUG_Y, color=f"{names[j]}", label="DA")
        j = j + 3

       
        ADV_C0_X = errs[
            element_percentage["g_augs"]
            + 2 : element_percentage["g_augs"]
            + element_percentage["eps"]
            + 2
        ]
        ADV_C0_Y = cics[
            element_percentage["g_augs"]
            + 2 : element_percentage["g_augs"]
            + element_percentage["eps"]
            + 2
        ]
        plt.plot(ADV_C0_X, ADV_C0_Y, color=f"{names[j]}", label="ADV, CLSDIV=0")
        j = j + 19

        ADV_C1_X = errs[
            element_percentage["g_augs"]
            + element_percentage["eps"]
            + 2 : element_percentage["g_augs"]
            + 2 * element_percentage["eps"]
            + 2
        ]
        ADV_C1_Y = cics[
            element_percentage["g_augs"]
            + element_percentage["eps"]
            + 2 : element_percentage["g_augs"]
            + 2 * element_percentage["eps"]
            + 2
        ]
        plt.plot(ADV_C1_X, ADV_C1_Y, color=f"{names[j]}", label="ADV, CLSDIV=1")
        j = j + 22

        # Naming the x-axis, y-axis and the whole graph
        plt.xlabel(f"ERR")
        plt.ylabel("CIC")
        plt.title(title)

        plt.xlim((12, 27))
        plt.ylim((0, 15))

        # Adding legend, which helps us recognize the curve according to it's color
        plt.legend()
        plt.grid(True)

        if save:
            plt.savefig(path)

        if plot:
            plt.show()

        # print(path)
        plt.close()

    def conversion(self):
        def get_name_column(lamp, culture, percent):
            ## Returns the name of tab and the columns
            if lamp:
                columns = {
                    "TS": [],
                    "ERR^LC": [],
                    "ERR^LC std": [],
                    "ERR^LF": [],
                    "ERR^LF std": [],
                    "ERR^LT": [],
                    "ERR^LT std": [],
                    "ERR": [],
                    "ERR std": [],
                    "W_ERR": [],
                    "W_ERR std": [],
                    "CIC": [],
                    "CIC std": [],
                }
                if culture == 0:
                    name = "LC"
                if culture == 1:
                    name = "LF"
                if culture == 2:
                    name = "LT"
            else:
                columns = {
                    "TS": [],
                    "ERR^CI": [],
                    "ERR^CI std": [],
                    "ERR^CJ": [],
                    "ERR^CJ std": [],
                    "ERR^CS": [],
                    "ERR^CS std": [],
                    "ERR": [],
                    "ERR std": [],
                    "W_ERR": [],
                    "W_ERR std": [],
                    "CIC": [],
                    "CIC std": [],
                }
                if culture == 0:
                    name = "CI"
                if culture == 1:
                    name = "CJ"
                if culture == 2:
                    name = "CS"

            name += f" with p_u={percent}"

            return name, columns

        # Structure of Tab is:
        # Per each DS
        # Per each Culture
        # Per each Percentage
        # Per each Data Augmentation in Training
        #   TestSet Mitigation  ERR^CULTURE 0 ERR^CULTURE 0 std,ERR^CULTURE 1,ERR^CULTURE 1 std,ERR^CULTURE 2,ERR^CULTURE 2 std,ERR,ERR std,W_ERR,W_ERR std,CIC,CIC std
        #   NOAUG   STD         value   value   value   value   value ..
        #   NOAUG   MIT         value   value   value   value   value ...
        #   AUG=..
        # ...
        # Using TAU=0.1, we have to select the right LAMBDA

        resacqobj = ResAcquisitionClass()

        standards = [1]
        alg = "DL"
        lamps = [0, 1]
        cultures = [0, 1, 2]
        percents = [0.2]
        augments = [0, 1]
        g_augments = {"0.2": np.logspace(-3, -1, 3)}
        adversary = [0, 1]
        epsilons = {
            "0.2": np.logspace(-3, -1, 3),
        }  
        imbalanceds = [0]
        diffusion = [0]
        g_diffusion = [0.0001]

        tot_elements_percentage = {
            "0.2": {"g_augs": len(g_augments["0.2"]), "eps": len(epsilons["0.2"])},
        }

        
        for lamp in lamps:
            for imb in imbalanceds:
                for culture in cultures:
                    for percent in percents:
                        # Here we have to divide the DataFrames
                        name, columns = get_name_column(lamp, culture, percent)

                        df = pd.DataFrame(columns=columns)
                        for adv in adversary:
                            for aug in augments:
                                if not aug:
                                    if not adv:
                                        pt = resacqobj.buildPath(
                                            basePath=basePath,
                                            standard=1,
                                            alg="DL",
                                            lamp=lamp,
                                            culture=culture,
                                            percent=percent,
                                            augment=aug,
                                            adversary=adv,
                                            lambda_index=0,
                                            taugment=0,
                                            tadversary=0,
                                            tgaug=0,
                                            teps=0,
                                            imbalanced=imb,
                                        )
                                        pt = pt.split("/")
                                        pt = pt[0 : len(pt) - 2]
                                        stdpt = ""
                                        for p in pt:
                                            stdpt += p + "/"
                                        stdpt += "res.csv"
                                        stddf = pd.read_csv(stdpt)
                                        trainName = "NOAUG"
                                        ls = self.convert2list(trainName, stddf)
                                        df.loc[len(df)] = ls
                                    else:
                                        for class_division in [0, 1]:
                                            for eps in epsilons[str(percent)]:

                                                pt = resacqobj.buildPath(
                                                    basePath=basePath,
                                                    standard=1,
                                                    alg="DL",
                                                    lamp=lamp,
                                                    culture=culture,
                                                    percent=percent,
                                                    augment=aug,
                                                    adversary=adv,
                                                    lambda_index=0,
                                                    taugment=0,
                                                    tadversary=0,
                                                    tgaug=0,
                                                    teps=0,
                                                    eps=eps,
                                                    class_division=class_division,
                                                    imbalanced=imb,
                                                )
                                                pt = pt.split("/")
                                                pt = pt[0 : len(pt) - 2]
                                                stdpt = ""
                                                for p in pt:
                                                    stdpt += p + "/"
                                                stdpt += "res.csv"
                                                stddf = pd.read_csv(stdpt)
                                                trainName = f"ADV, CLSDIV={class_division}, eps={eps}"
                                                ls = self.convert2list(trainName, stddf)
                                                df.loc[len(df)] = ls
                                else:
                                    if not adv:
                                        for dif in diffusion:
                                            if not dif:
                                                for g_augment in g_augments[
                                                    str(percent)
                                                ]:
                                                    pt = resacqobj.buildPath(
                                                        basePath=basePath,
                                                        standard=1,
                                                        alg="DL",
                                                        lamp=lamp,
                                                        culture=culture,
                                                        percent=percent,
                                                        augment=aug,
                                                        adversary=adv,
                                                        lambda_index=0,
                                                        taugment=0,
                                                        tadversary=0,
                                                        tgaug=0,
                                                        teps=0,
                                                        g_augment=g_augment,
                                                        eps=0,
                                                        class_division=0,
                                                        imbalanced=imb,
                                                        diffusion=dif,
                                                    )
                                                    pt = pt.split("/")
                                                    pt = pt[0 : len(pt) - 2]
                                                    stdpt = ""
                                                    for p in pt:
                                                        stdpt += p + "/"
                                                    stdpt += "res.csv"
                                                    stddf = pd.read_csv(stdpt)
                                                    trainName = f"AUG, g={g_augment}"
                                                    df.loc[len(df)] = self.convert2list(
                                                        trainName, stddf
                                                    )
                                            else:
                                                g_augment = g_diffusion[0]
                                                pt = resacqobj.buildPath(
                                                    basePath=basePath,
                                                    standard=1,
                                                    alg="DL",
                                                    lamp=lamp,
                                                    culture=culture,
                                                    percent=percent,
                                                    augment=aug,
                                                    adversary=adv,
                                                    lambda_index=0,
                                                    g_augment=g_augment,
                                                    taugment=0,
                                                    tadversary=0,
                                                    tgaug=0,
                                                    teps=0,
                                                    imbalanced=imb,
                                                    diffusion=dif,
                                                )
                                                pt = pt.split("/")
                                                pt = pt[0 : len(pt) - 2]
                                                stdpt = ""
                                                for p in pt:
                                                    stdpt += p + "/"
                                                stdpt += "res.csv"
                                                stddf = pd.read_csv(stdpt)
                                                trainName = (
                                                    f"DIFF with AUG, g={g_augment}"
                                                )
                                                ls = self.convert2list(trainName, stddf)
                                                df.loc[len(df)] = ls

                        print(name)
                        df.style.set_properties(
                            subset=["Total"],
                            **{"font-weight": "bold", "border-left": "1pt solid black"},
                        ).apply(functools.partial(self.borderTop, "STD"), axis=1)
                        print(df.to_string())
                        pt = self.get_path(
                            "../Results/TABRES/", lamp, culture, percent, imb
                        )
                        fileObj = FileManagerClass(pt)
                        df.to_csv(pt + "df")
                        pt = self.get_path(
                            "../Results/HTML/", lamp, culture, percent, imb
                        )
                        fileObj = FileManagerClass(pt)
                        df.to_html(pt + "res.html")

                        if lamp:
                            if culture == 0:
                                title = f"Chinese"
                            if culture == 1:
                                title = f"French"
                            if culture == 2:
                                title = f"Turkish"
                        else:
                            if culture == 0:
                                title = f"Indian"
                            if culture == 1:
                                title = f"Japanese"
                            if culture == 2:
                                title = f"Scandinavian"
                        if imb:
                            ovs = "OS"
                        else:
                            ovs = "no OS"
                        title += f", {ovs}, percent={percent}"
                        self.visualize(
                            df,
                            title=title,
                            element_percentage=tot_elements_percentage[str(percent)],
                        )


def main():

    res2tabObj = Res2TabClass()
    res2tabObj.conversion()


if __name__ == "__main__":
    main()
