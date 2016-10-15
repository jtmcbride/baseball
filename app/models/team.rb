class Team < ActiveRecord::Base
  has_many :battings, foreign_key: 'fk_teamid', primary_key: 'pk_teamid'
  has_many :players, through: :battings
end
