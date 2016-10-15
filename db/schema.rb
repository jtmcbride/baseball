# encoding: UTF-8
# This file is auto-generated from the current state of the database. Instead
# of editing this file, please use the migrations feature of Active Record to
# incrementally modify your database, and then regenerate this schema definition.
#
# Note that this schema.rb definition is the authoritative source for your
# database schema. If you need to create the application database on another
# system, you should be using db:schema:load, not running all the migrations
# from scratch. The latter is a flawed and unsustainable approach (the more migrations
# you'll amass, the slower it'll run and the greater likelihood for issues).
#
# It's strongly recommended that you check this file into your version control system.

ActiveRecord::Schema.define(version: 20161002174658) do

  # These are extensions that must be enabled in order to support this database
  enable_extension "plpgsql"

  create_table "babe_ruth_numbers", force: :cascade do |t|
    t.string  "player_id", null: false
    t.string  "team_id",   null: false
    t.integer "distance",  null: false
  end

  add_index "babe_ruth_numbers", ["team_id"], name: "index_babe_ruth_numbers_on_team_id", unique: true, using: :btree

  create_table "batting", force: :cascade do |t|
    t.string  "playerid",  limit: 9
    t.integer "yearid"
    t.integer "stint"
    t.string  "teamid",    limit: 3
    t.string  "lgid",      limit: 2
    t.integer "g"
    t.integer "ab"
    t.integer "r"
    t.integer "h"
    t.integer "dub"
    t.integer "trip"
    t.integer "hr"
    t.integer "rbi"
    t.integer "sb"
    t.integer "cs"
    t.integer "bb"
    t.integer "so"
    t.integer "ibb"
    t.integer "hbp"
    t.integer "sh"
    t.integer "sf"
    t.integer "gidp"
    t.string  "fk_teamid", limit: 20
  end

  add_index "batting", ["fk_teamid"], name: "fk_teamid_index", using: :btree
  add_index "batting", ["playerid"], name: "index_playerid", using: :btree

  create_table "master", primary_key: "playerid", force: :cascade do |t|
    t.integer "birthyear"
    t.integer "birthmonth"
    t.integer "birthday"
    t.text    "birthcountry"
    t.text    "birthstate"
    t.text    "birthcity"
    t.integer "deathyear"
    t.integer "deathmonth"
    t.integer "deathday"
    t.text    "deathcountry"
    t.text    "deathstate"
    t.text    "deathcity"
    t.text    "namefirst"
    t.text    "namelast"
    t.text    "namegiven"
    t.integer "weight"
    t.integer "height"
    t.string  "bats",         limit: 1
    t.string  "throws",       limit: 1
    t.date    "debut"
    t.date    "finalgame"
    t.string  "retroid",      limit: 9
    t.string  "bbrefid",      limit: 9
  end

  create_table "teams", force: :cascade do |t|
    t.integer "yearid"
    t.string  "lgid"
    t.string  "teamid"
    t.string  "franchid"
    t.string  "divid"
    t.integer "rank"
    t.integer "g"
    t.integer "ghome"
    t.integer "w"
    t.integer "l"
    t.string  "divwin"
    t.string  "wcwin"
    t.string  "lgwin"
    t.string  "wswin"
    t.integer "r"
    t.integer "ab"
    t.integer "h"
    t.integer "dub"
    t.integer "trip"
    t.integer "hr"
    t.integer "bb"
    t.integer "so"
    t.integer "sb"
    t.integer "cs"
    t.integer "hbp"
    t.integer "sf"
    t.integer "ra"
    t.integer "er"
    t.float   "era"
    t.integer "cg"
    t.integer "sho"
    t.integer "sv"
    t.integer "ipouts"
    t.integer "ha"
    t.integer "hra"
    t.integer "bba"
    t.integer "soa"
    t.integer "e"
    t.integer "dp"
    t.float   "fp"
    t.string  "name"
    t.string  "park"
    t.integer "attendance"
    t.integer "bpf"
    t.integer "ppf"
    t.string  "teamidbr"
    t.string  "teamidlahman45"
    t.string  "teamidretro"
    t.string  "pk_teamid",      limit: 20
    t.string  "fk_teamid",      limit: 20
  end

  add_index "teams", ["pk_teamid"], name: "teamid_index", using: :btree
  add_index "teams", ["pk_teamid"], name: "teams_pk_teamid_key", unique: true, using: :btree

  add_foreign_key "batting", "master", column: "playerid", primary_key: "playerid", name: "batting_playerid_fkey"
  add_foreign_key "batting", "teams", column: "fk_teamid", primary_key: "pk_teamid", name: "batting_fk_teamid_fkey"
end
